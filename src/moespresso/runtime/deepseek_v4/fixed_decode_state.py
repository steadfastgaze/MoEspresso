"""Fixed-shape decode cache state for DS4 compressed-attention layers.

The stock `DeepseekV4Cache` grows its compressor and indexer state by
concatenation: the partial-window accumulators cycle through `ratio`
shapes, the pooled pool gains one row every `ratio`-th token, and the
`pooled_fp8` / `pooled_qat` derived caches reconcatenate on every new
row. Every one of those growth events is a per-token shape change, which
forces a whole-layer compiled decode island to retrace. This module
replaces the storage layout and the append mechanism on the single-token
decode path with preallocated capacity buffers plus valid-row counts,
while keeping the arithmetic per row byte-for-byte the stock op sequence.

Contract
--------
- Window accumulators become fixed capacity buffers written by slice
  update: `2 * ratio` rows for the overlap layout (previous window in
  rows `[0, ratio)`, partial window in rows `[ratio, 2 * ratio)`) and
  `ratio` rows for the plain layout. On window completion the overlap
  buffer copies its second half over its first half. Valid-row counts carry the
  fill state while shapes stay fixed.
- The pooled pool and the `pooled_fp8` / `pooled_qat` derived caches
  become capacity buffers with `ceil(max_context / ratio)` rows, where
  `max_context` is the serve-configured maximum context
  (`MOESPRESSO_DSV4_DECODE_MAX_CONTEXT`, default 4096). Appends are
  row-index slice updates.
- Row counts live as Python mirrors on the branch state and enter the
  graph through a fresh per-token params array
  (`decode_step_params`), never as baked constants inside a trace.
- Indexer scores computed over the capacity buffer pad the invalid tail
  with negative infinity (`pad_scores_to_capacity`). Top-k selection
  still runs on the valid-prefix slice, whose values are bit-identical
  to the concat-layout scores, so the selection and its order match the
  stock path exactly. Selecting over the padded row instead is unchanged
  only when the boundary is tie-free; a parity test pins both forms.
- The legacy dict keys (`buffer_kv`, `buffer_gate`, `pooled`,
  `pooled_fp8`, `pooled_qat`, ...) are republished as valid-row views
  after every fixed-path mutation, so every reader of the stock layout
  keeps seeing correct values.

Fail-closed rules
-----------------
Eligibility is per call and delegates before any legacy-state mutation:
multi-token chunks, batch sizes other than one, overlap accumulation at
offset zero, dtype mismatches, capacity exhaustion, and any external
replacement of a legacy dict value (detected by object-identity mirrors)
all drop the fixed representation and fall back to the stock concat
path, which remains authoritative. `MOESPRESSO_DSV4_DECODE_FIXED_STATE=0`
is the kill switch: the cache wrapper becomes a no-op and the stock
contract is untouched.

Engagement
----------
`fixed_state_decode_layers` (surfaced per layer through the attention
layer stats) counts decode steps whose compressor state, pool append,
and fp8 cache all ran on the fixed path. Sliding-window layers keep no
compressed pool state, so the counter covers the 41 compressed layers;
a full-engagement decode token reads 41.

Prefix reuse stays in-memory: capacity buffers are process memory and
are never serialized. Trim rewinds row counts and clears the derived
caches without reallocating.
"""

from __future__ import annotations

import math
import os
from types import MethodType

_FIXED_STATE_ENV = "MOESPRESSO_DSV4_DECODE_FIXED_STATE"
_MAX_CONTEXT_ENV = "MOESPRESSO_DSV4_DECODE_MAX_CONTEXT"
_DEFAULT_MAX_CONTEXT = 4096

_STATE_KEYS = ("compressor_state", "indexer_state")
_LEGACY_KEYS = ("buffer_kv", "buffer_gate", "pooled")
_AUX_KEYS_BY_STATE = {
    "compressor_state": ("pooled_fp8", "pooled_fp8_rows"),
    "indexer_state": ("pooled_qat", "pooled_qat_rows"),
}

_BRANCHES_ATTR = "_moespresso_dsv4_fixed_branches"
_ORIGINALS_ATTR = "_moespresso_dsv4_fixed_originals"
_INSTALLED_ATTR = "_moespresso_dsv4_fixed_decode_state"
_MAX_CONTEXT_ATTR = "_moespresso_dsv4_fixed_max_context"


def fixed_decode_state_enabled() -> bool:
    """Kill switch: `MOESPRESSO_DSV4_DECODE_FIXED_STATE=0` restores the
    stock concat-growth cache contract exactly."""
    return os.environ.get(_FIXED_STATE_ENV, "1") != "0"


def fixed_decode_max_context() -> int:
    """Serve-configured maximum context that sizes the capacity buffers."""
    return int(os.environ.get(_MAX_CONTEXT_ENV, str(_DEFAULT_MAX_CONTEXT)))


class _FixedAux:
    """Fixed-capacity derived cache (fp8 or QAT rows) for one branch."""

    __slots__ = ("cap", "rows", "view")

    def __init__(self) -> None:
        self.cap = None
        self.rows = 0
        self.view = None


class _FixedBranchState:
    """Capacity buffers plus row counts for one compressor state dict."""

    __slots__ = (
        "ratio",
        "overlap",
        "head_dim",
        "pool_capacity",
        "win_kv",
        "win_gate",
        "prev_rows",
        "partial_rows",
        "pool_cap",
        "pool_rows",
        "mirrors",
        "aux",
        "pending_pos",
        "engaged_pos",
    )

    def __init__(self, *, ratio: int, overlap: bool, head_dim: int,
                 pool_capacity: int) -> None:
        self.ratio = int(ratio)
        self.overlap = bool(overlap)
        self.head_dim = int(head_dim)
        self.pool_capacity = int(pool_capacity)
        self.win_kv = None
        self.win_gate = None
        self.prev_rows = 0
        self.partial_rows = 0
        self.pool_cap = None
        self.pool_rows = 0
        self.mirrors: dict = {}
        self.aux: dict[str, _FixedAux] = {}
        self.pending_pos: int | None = None
        self.engaged_pos: int | None = None


def _state_dict(cache, state_key: str):
    state = getattr(cache, state_key, None)
    return state if isinstance(state, dict) else None


def _branch_valid(branch: _FixedBranchState, state: dict) -> bool:
    return all(state.get(key) is branch.mirrors.get(key) for key in _LEGACY_KEYS)


def _drop_branch(cache, state_key: str) -> None:
    branches = getattr(cache, _BRANCHES_ATTR, None)
    if isinstance(branches, dict):
        branches.pop(state_key, None)


def _get_valid_branch(cache, state_key: str) -> _FixedBranchState | None:
    branches = getattr(cache, _BRANCHES_ATTR, None)
    if not isinstance(branches, dict):
        return None
    branch = branches.get(state_key)
    if branch is None:
        return None
    state = _state_dict(cache, state_key)
    if state is None or not _branch_valid(branch, state):
        # A legacy key was replaced behind the fixed layout. The dict is
        # authoritative; drop the fixed representation and readopt later.
        branches.pop(state_key, None)
        return None
    return branch


def engaged_branch(cache, state_key: str, start_pos) -> _FixedBranchState | None:
    """Branch whose window and pool updates ran fixed at this position."""
    branch = _get_valid_branch(cache, state_key)
    if branch is None or branch.engaged_pos != int(start_pos):
        return None
    return branch


def decode_step_params(mx, *, offset: int, pool_rows: int):
    """Per-token params array carrying position and valid-row count.

    Per-token scalars ride fresh small arrays so a later compiled island
    reads them as data instead of baking them into a trace.
    """
    return mx.array([int(offset), int(pool_rows)], dtype=mx.int32)


def pad_scores_to_capacity(mx, scores, params):
    """Mask score columns at and past the valid-row count to -inf.

    `scores` covers the full pool capacity; `params[1]` is the valid-row
    count. Valid columns pass through bitwise (`mx.where` selects), so
    the valid prefix of the padded row equals the concat-layout scores.
    """
    capacity = int(scores.shape[-1])
    valid = (mx.arange(capacity, dtype=mx.int32) < params[1]).reshape(
        (1,) * (scores.ndim - 1) + (capacity,)
    )
    return mx.where(valid, scores, mx.array(-mx.inf, dtype=scores.dtype))


def aux_capacity_buffer(branch: _FixedBranchState, name: str):
    """Full capacity buffer of a derived cache, or None before adoption."""
    aux = branch.aux.get(name)
    return None if aux is None else aux.cap


def aux_fixed_update(branch: _FixedBranchState, state: dict, *, name: str,
                     rows_key: str, pooled, transform):
    """Append-transform new pool rows into a fixed-capacity derived cache.

    `pooled` is the valid-row pool view and `transform` is the stock op
    sequence applied to new rows (fp8 round trip or indexer QAT). Returns
    `(valid_view, prior_rows)` after publishing the legacy keys, or None
    when the fixed path cannot proceed; the caller then falls back to the
    stock concat path, which trusts the legacy keys as they stand.
    """
    import mlx.core as mx

    rows = int(pooled.shape[1])
    if rows > branch.pool_capacity:
        return None
    aux = branch.aux.get(name)
    cached_obj = state.get(name)
    cached_rows = int(state.get(rows_key, 0) or 0)
    if (
        aux is None
        or cached_obj is not aux.view
        or cached_rows != aux.rows
    ):
        # Adopt from the legacy keys (authoritative). A malformed cache
        # (width disagreeing with its row count) stays on the stock path.
        aux = _FixedAux()
        if cached_obj is not None and cached_rows > 0:
            if (
                cached_rows > branch.pool_capacity
                or cached_rows > rows
                or int(cached_obj.shape[1]) != cached_rows
            ):
                return None
            aux.cap = mx.zeros(
                (int(cached_obj.shape[0]), branch.pool_capacity,
                 int(cached_obj.shape[-1])),
                dtype=cached_obj.dtype,
            )
            aux.cap[:, :cached_rows] = cached_obj
            aux.rows = cached_rows
        branch.aux[name] = aux

    prior_rows = aux.rows
    if rows < aux.rows:
        # The pool shrank without a trim reset; recompute from scratch.
        aux.rows = 0
        prior_rows = 0
    if rows > aux.rows:
        tail = transform(pooled[:, aux.rows:, :])
        if aux.cap is None:
            aux.cap = mx.zeros(
                (int(tail.shape[0]), branch.pool_capacity,
                 int(tail.shape[-1])),
                dtype=tail.dtype,
            )
        elif tail.dtype != aux.cap.dtype:
            branch.aux.pop(name, None)
            return None
        if int(tail.shape[1]):
            aux.cap[:, aux.rows:rows] = tail
        aux.rows = rows
    if aux.cap is None:
        aux.cap = mx.zeros(
            (int(pooled.shape[0]), branch.pool_capacity,
             int(pooled.shape[-1])),
            dtype=pooled.dtype,
        )
    view = aux.cap[:, :rows]
    aux.view = view
    state[name] = view
    state[rows_key] = rows
    return view, prior_rows


def _pool_capacity(cache, ratio: int) -> int:
    max_context = int(getattr(cache, _MAX_CONTEXT_ATTR, 0) or 0)
    if max_context <= 0 or ratio <= 0:
        return 0
    return int(math.ceil(max_context / ratio))


def _adopt_window(branch: _FixedBranchState, state: dict, *, mx, out_dim: int) -> bool:
    """Copy the legacy window accumulators into the capacity layout."""
    buf_kv = state.get("buffer_kv")
    buf_gate = state.get("buffer_gate")
    kv_rows = int(buf_kv.shape[1]) if buf_kv is not None else 0
    gate_rows = int(buf_gate.shape[1]) if buf_gate is not None else 0
    if kv_rows != gate_rows:
        return False
    ratio = branch.ratio
    win_rows = 2 * ratio if branch.overlap else ratio
    if branch.overlap:
        if kv_rows >= ratio:
            prev_rows, partial = ratio, kv_rows - ratio
        else:
            prev_rows, partial = 0, kv_rows
        if partial >= ratio:
            return False
    else:
        prev_rows, partial = 0, kv_rows
        if partial >= ratio:
            return False
    if kv_rows:
        if int(buf_kv.shape[-1]) != out_dim or int(buf_gate.shape[-1]) != out_dim:
            return False
        batch = int(buf_kv.shape[0])
        branch.win_kv = mx.zeros((batch, win_rows, out_dim), dtype=buf_kv.dtype)
        branch.win_gate = mx.zeros((batch, win_rows, out_dim), dtype=buf_gate.dtype)
        if branch.overlap:
            if prev_rows:
                branch.win_kv[:, :ratio] = buf_kv[:, :ratio]
                branch.win_gate[:, :ratio] = buf_gate[:, :ratio]
            if partial:
                branch.win_kv[:, ratio:ratio + partial] = buf_kv[:, prev_rows:]
                branch.win_gate[:, ratio:ratio + partial] = buf_gate[:, prev_rows:]
        elif partial:
            branch.win_kv[:, :partial] = buf_kv
            branch.win_gate[:, :partial] = buf_gate
    branch.prev_rows = prev_rows
    branch.partial_rows = partial
    return True


def _ensure_branch(cache, state, state_key: str, *, mx, ratio: int,
                   overlap: bool, head_dim: int, out_dim: int):
    """Validate or adopt the fixed branch for one decode-shaped call."""
    branches = getattr(cache, _BRANCHES_ATTR, None)
    if not isinstance(branches, dict):
        return None
    branch = branches.get(state_key)
    if branch is not None:
        if (
            _branch_valid(branch, state)
            and branch.ratio == ratio
            and branch.overlap == overlap
            and branch.head_dim == head_dim
        ):
            return branch
        branches.pop(state_key, None)
    capacity = _pool_capacity(cache, ratio)
    if capacity <= 0:
        return None
    pooled = state.get("pooled")
    if pooled is not None and int(pooled.shape[1]) > capacity:
        return None
    branch = _FixedBranchState(
        ratio=ratio, overlap=overlap, head_dim=head_dim, pool_capacity=capacity,
    )
    if not _adopt_window(branch, state, mx=mx, out_dim=out_dim):
        return None
    # Trust the current legacy objects as the adoption source; the pool
    # itself is copied lazily by the update_pool override.
    for key in _LEGACY_KEYS:
        branch.mirrors[key] = state.get(key)
    branches[state_key] = branch
    return branch


def _publish_window(branch: _FixedBranchState, state: dict) -> None:
    ratio = branch.ratio
    if branch.overlap:
        if branch.prev_rows:
            kv_view = branch.win_kv[:, :ratio + branch.partial_rows]
            gate_view = branch.win_gate[:, :ratio + branch.partial_rows]
        else:
            kv_view = branch.win_kv[:, ratio:ratio + branch.partial_rows]
            gate_view = branch.win_gate[:, ratio:ratio + branch.partial_rows]
    else:
        kv_view = branch.win_kv[:, :branch.partial_rows]
        gate_view = branch.win_gate[:, :branch.partial_rows]
    state["buffer_kv"] = kv_view
    state["buffer_gate"] = gate_view
    branch.mirrors["buffer_kv"] = kv_view
    branch.mirrors["buffer_gate"] = gate_view


def _decode_shaped(kv, gate) -> bool:
    return (
        len(kv.shape) == 3
        and len(gate.shape) == 3
        and int(kv.shape[0]) == 1
        and int(kv.shape[1]) == 1
        and int(gate.shape[0]) == 1
        and int(gate.shape[1]) == 1
    )


def _fixed_accumulate_windows(self, kv, gate, state_key, ratio, start_pos):
    import mlx.core as mx

    originals = getattr(self, _ORIGINALS_ATTR)
    state = _state_dict(self, state_key)
    ratio = int(ratio)
    if state is None or ratio <= 0 or not _decode_shaped(kv, gate):
        _drop_branch(self, state_key)
        return originals["accumulate_windows"](kv, gate, state_key, ratio, start_pos)
    out_dim = int(kv.shape[-1])
    branch = _ensure_branch(
        self, state, state_key, mx=mx,
        ratio=ratio, overlap=False, head_dim=out_dim, out_dim=out_dim,
    )
    if branch is None:
        return originals["accumulate_windows"](kv, gate, state_key, ratio, start_pos)
    if branch.win_kv is None:
        batch = int(kv.shape[0])
        branch.win_kv = mx.zeros((batch, ratio, out_dim), dtype=kv.dtype)
        branch.win_gate = mx.zeros((batch, ratio, out_dim), dtype=gate.dtype)
    if (
        kv.dtype != branch.win_kv.dtype
        or gate.dtype != branch.win_gate.dtype
        or out_dim != int(branch.win_kv.shape[-1])
    ):
        _drop_branch(self, state_key)
        return originals["accumulate_windows"](kv, gate, state_key, ratio, start_pos)

    prior = branch.partial_rows
    branch.win_kv[:, prior:prior + 1] = kv
    branch.win_gate[:, prior:prior + 1] = gate
    partial = prior + 1
    pool_base = max(0, int(start_pos)) - prior
    if partial == ratio:
        ready_kv = branch.win_kv[:, :ratio]
        ready_gate = branch.win_gate[:, :ratio]
        branch.partial_rows = 0
    else:
        ready_kv = branch.win_kv[:, :0]
        ready_gate = branch.win_gate[:, :0]
        branch.partial_rows = partial
    _publish_window(branch, state)
    branch.pending_pos = int(start_pos)
    branch.engaged_pos = None
    return ready_kv, ready_gate, pool_base


def _fixed_accumulate_overlap_windows(self, kv, gate, state_key, ratio,
                                      start_pos, head_dim):
    import mlx.core as mx

    originals = getattr(self, _ORIGINALS_ATTR)
    state = _state_dict(self, state_key)
    ratio = int(ratio)
    head_dim = int(head_dim)
    if (
        state is None
        or ratio <= 0
        or int(start_pos) == 0
        or not _decode_shaped(kv, gate)
    ):
        # The stock method resets the accumulators at offset zero; keep
        # that fresh-sequence semantics on the stock path.
        _drop_branch(self, state_key)
        return originals["accumulate_overlap_windows"](
            kv, gate, state_key, ratio, start_pos, head_dim)
    out_dim = int(kv.shape[-1])
    branch = _ensure_branch(
        self, state, state_key, mx=mx,
        ratio=ratio, overlap=True, head_dim=head_dim, out_dim=out_dim,
    )
    if branch is None:
        return originals["accumulate_overlap_windows"](
            kv, gate, state_key, ratio, start_pos, head_dim)
    if branch.win_kv is None:
        batch = int(kv.shape[0])
        branch.win_kv = mx.zeros((batch, 2 * ratio, out_dim), dtype=kv.dtype)
        branch.win_gate = mx.zeros((batch, 2 * ratio, out_dim), dtype=gate.dtype)
    if (
        kv.dtype != branch.win_kv.dtype
        or gate.dtype != branch.win_gate.dtype
        or out_dim != int(branch.win_kv.shape[-1])
    ):
        _drop_branch(self, state_key)
        return originals["accumulate_overlap_windows"](
            kv, gate, state_key, ratio, start_pos, head_dim)

    batch = int(kv.shape[0])
    prior = branch.partial_rows
    idx = ratio + prior
    branch.win_kv[:, idx:idx + 1] = kv
    branch.win_gate[:, idx:idx + 1] = gate
    partial = prior + 1
    pool_base = max(0, int(start_pos) - prior)
    if partial == ratio:
        # Same construction as the stock `_make_row`: zero / -inf filled
        # rows taking previous-window first-half features and
        # current-window second-half features.
        row_kv = mx.zeros((batch, 1, 2 * ratio, head_dim), dtype=kv.dtype)
        row_gate = mx.full(
            (batch, 1, 2 * ratio, head_dim), -float("inf"), dtype=gate.dtype)
        if branch.prev_rows == ratio:
            row_kv[:, 0, :ratio] = branch.win_kv[:, :ratio, :head_dim]
            row_gate[:, 0, :ratio] = branch.win_gate[:, :ratio, :head_dim]
        row_kv[:, 0, ratio:] = branch.win_kv[:, ratio:, head_dim:]
        row_gate[:, 0, ratio:] = branch.win_gate[:, ratio:, head_dim:]
        branch.win_kv[:, :ratio] = branch.win_kv[:, ratio:]
        branch.win_gate[:, :ratio] = branch.win_gate[:, ratio:]
        branch.prev_rows = ratio
        branch.partial_rows = 0
        ret = (row_kv, row_gate, pool_base)
    else:
        branch.partial_rows = partial
        ret = (
            mx.zeros((batch, 0, 2 * ratio, head_dim), dtype=kv.dtype),
            mx.zeros((batch, 0, 2 * ratio, head_dim), dtype=gate.dtype),
            pool_base,
        )
    _publish_window(branch, state)
    branch.pending_pos = int(start_pos)
    branch.engaged_pos = None
    return ret


def _fixed_update_pool(self, new_pooled, state_key):
    import mlx.core as mx

    originals = getattr(self, _ORIGINALS_ATTR)
    branches = getattr(self, _BRANCHES_ATTR, None)
    branch = branches.get(state_key) if isinstance(branches, dict) else None
    if branch is None or branch.pending_pos is None:
        return originals["update_pool"](new_pooled, state_key)
    pos = branch.pending_pos
    branch.pending_pos = None
    state = _state_dict(self, state_key)
    rows_new = int(new_pooled.shape[1])
    if (
        state is None
        or not _branch_valid(branch, state)
        or int(new_pooled.shape[0]) != 1
        or rows_new > 1
    ):
        _drop_branch(self, state_key)
        return originals["update_pool"](new_pooled, state_key)

    if branch.pool_cap is None:
        legacy_pool = state.get("pooled")
        prior_rows = int(legacy_pool.shape[1]) if legacy_pool is not None else 0
        if prior_rows + rows_new > branch.pool_capacity:
            _drop_branch(self, state_key)
            return originals["update_pool"](new_pooled, state_key)
        source = legacy_pool if legacy_pool is not None else new_pooled
        branch.pool_cap = mx.zeros(
            (int(source.shape[0]), branch.pool_capacity, int(source.shape[-1])),
            dtype=source.dtype,
        )
        if prior_rows:
            branch.pool_cap[:, :prior_rows] = legacy_pool
        branch.pool_rows = prior_rows
    if rows_new:
        if (
            new_pooled.dtype != branch.pool_cap.dtype
            or int(new_pooled.shape[-1]) != int(branch.pool_cap.shape[-1])
            or branch.pool_rows + rows_new > branch.pool_capacity
        ):
            _drop_branch(self, state_key)
            return originals["update_pool"](new_pooled, state_key)
        branch.pool_cap[:, branch.pool_rows:branch.pool_rows + rows_new] = new_pooled
        branch.pool_rows += rows_new
    if branch.pool_rows:
        view = branch.pool_cap[:, :branch.pool_rows]
        state["pooled"] = view
        branch.mirrors["pooled"] = view
        result = view
    else:
        state["pooled"] = None
        branch.mirrors["pooled"] = None
        result = mx.zeros(
            (int(new_pooled.shape[0]), 0, int(new_pooled.shape[-1])),
            dtype=new_pooled.dtype,
        )
    branch.engaged_pos = pos
    return result


def _fixed_trim(self, n):
    branches = getattr(self, _BRANCHES_ATTR, None)
    live: dict[str, _FixedBranchState] = {}
    if isinstance(branches, dict):
        for state_key in _STATE_KEYS:
            branch = branches.get(state_key)
            state = _state_dict(self, state_key)
            usable = (
                branch is not None
                and state is not None
                and _branch_valid(branch, state)
                # A branch whose pool never reached the capacity buffer
                # cannot rewind the pool by counts; the stock trim owns it.
                and (branch.pool_cap is not None
                     or branch.mirrors.get("pooled") is None)
            )
            if usable:
                live[state_key] = branch
            elif branch is not None:
                branches.pop(state_key, None)
    if not live:
        return getattr(self, _ORIGINALS_ATTR)["trim"](n)

    # Fixed-layout trim: rewind row counts and clear the derived caches
    # without reallocating, replicating the stock trim's proportional
    # pool-row drop and unconditional partial-window clear.
    rv = self.local.trim(n)
    n = int(n)
    for state_key in _STATE_KEYS:
        state = _state_dict(self, state_key)
        if state is None:
            continue
        branch = live.get(state_key)
        if branch is None:
            state["buffer_kv"] = None
            state["buffer_gate"] = None
            ratio = self.compress_ratio
            if ratio is None or ratio <= 0:
                state["pooled"] = None
            else:
                rows_to_drop = max(1, n // int(ratio)) if n > 0 else 0
                pooled = state.get("pooled")
                if rows_to_drop and pooled is not None:
                    keep = max(0, int(pooled.shape[1]) - rows_to_drop)
                    state["pooled"] = pooled[:, :keep, :] if keep else None
        else:
            branch.prev_rows = 0
            branch.partial_rows = 0
            state["buffer_kv"] = None
            state["buffer_gate"] = None
            branch.mirrors["buffer_kv"] = None
            branch.mirrors["buffer_gate"] = None
            rows_to_drop = max(1, n // branch.ratio) if n > 0 else 0
            if rows_to_drop:
                branch.pool_rows = max(0, branch.pool_rows - rows_to_drop)
            if branch.pool_rows and branch.pool_cap is not None:
                view = branch.pool_cap[:, :branch.pool_rows]
            else:
                branch.pool_rows = 0
                view = None
            state["pooled"] = view
            branch.mirrors["pooled"] = view
            branch.aux.clear()
            branch.pending_pos = None
            branch.engaged_pos = None
        for key in _AUX_KEYS_BY_STATE[state_key]:
            state.pop(key, None)
    return rv


def install_fixed_decode_state(cache):
    """Install the fixed-shape decode state overrides on a DS4 cache.

    Returns the cache unchanged when the kill switch is set, when the
    capacity configuration is unusable, or when the cache does not carry
    the expected composite-state surface.
    """
    if not fixed_decode_state_enabled():
        return cache
    if getattr(cache, _INSTALLED_ATTR, False):
        return cache
    max_context = fixed_decode_max_context()
    if max_context <= 0:
        return cache
    required = (
        "accumulate_windows",
        "accumulate_overlap_windows",
        "update_pool",
        "trim",
        "local",
        "compressor_state",
        "indexer_state",
    )
    if any(getattr(cache, name, None) is None for name in required):
        return cache

    # Originals live in an instance dict so `copy.deepcopy` (the prefix
    # cache fork) rebinds them to the copied cache through the deepcopy
    # memo. A closure over a bound method would keep pointing at the
    # source cache after a fork.
    originals = {
        "accumulate_windows": cache.accumulate_windows,
        "accumulate_overlap_windows": cache.accumulate_overlap_windows,
        "update_pool": cache.update_pool,
        "trim": cache.trim,
    }
    object.__setattr__(cache, _ORIGINALS_ATTR, originals)
    object.__setattr__(cache, _BRANCHES_ATTR, {})
    object.__setattr__(cache, _MAX_CONTEXT_ATTR, int(max_context))
    object.__setattr__(
        cache, "accumulate_windows", MethodType(_fixed_accumulate_windows, cache))
    object.__setattr__(
        cache,
        "accumulate_overlap_windows",
        MethodType(_fixed_accumulate_overlap_windows, cache),
    )
    object.__setattr__(cache, "update_pool", MethodType(_fixed_update_pool, cache))
    object.__setattr__(cache, "trim", MethodType(_fixed_trim, cache))
    object.__setattr__(cache, _INSTALLED_ATTR, True)
    return cache
