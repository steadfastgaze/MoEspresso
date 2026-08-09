"""Bit-exact cache rollback for DSpark speculative verification.

A verify forward advances the target caches by the whole draft block, and
rejection must rewind them to the accepted frontier. `trim_prompt_cache`
cannot do that exactly on the DeepSeek-V4 hybrid caches: the composite
`DeepseekV4Cache.trim` clears the compressor and indexer partial-window
buffers whose source tokens are never re-fed, so the next pooled row is
computed from an incomplete window, and the rotating local window is only
trimmable before it wraps. This module replaces trimming with a snapshot
and restore pair built around the mutations the verify forward actually
performs.

Contract
--------
`capture_verify_state` runs before a verify forward of `n_tokens` rows and
records, per layer cache, every location that forward can mutate:

- plain `KVCache`: the offset and copies of the key and value rows in
  `[offset, offset + n)` that are already allocated. The forward writes
  those rows in place.
- composite `DeepseekV4Cache`: copies of the rotating local window buffers
  with their offset and ring index, the compressor and indexer
  partial-window buffers, the pooled row counts, and references to the
  derived `pooled_fp8` and `pooled_qat` caches with their row counts.
  Transient recorders wrap `local.update_and_fetch` and the two
  `accumulate_*` methods for the duration of the forward, capturing the
  per-position rows the forward feeds into the cache. Those rows are what
  make an exact rewind possible: the partial-window buffers after a
  rollback must contain per-position projections of kept tokens that the
  forward may have already consumed into pooled rows, and the post-forward
  buffers alone no longer hold them.

`restore_verify_state(cache, snapshot, keep_tokens)` rewinds every layer so
the cache holds exactly the first `keep_tokens` positions of the verify
block. Kept locations keep the values the verify forward wrote; everything
at positions `>= offset + keep_tokens` returns to its pre-verify content:

- local window: the pre-verify buffers are reinstalled and the kept rows
  are replayed through the stock single-row ring update. The resulting
  layout is exactly the layout stepwise decoding produces, so the cache
  physical state stays a pure function of position rather than of the
  accept history, and it works at any offset, wrapped or not.
- partial-window buffers: rebuilt as the positional tail of the pre-verify
  buffer concatenated with the kept recorded rows. The stock accumulators
  are per-slot (each buffered row is one position's projection), so the
  kept-prefix buffer content is exactly this tail.
- pooled pools: sliced to the pre-verify row count plus the windows that
  complete within the kept prefix. Appends preserve the existing prefix
  bitwise, so the slice equals the pool a kept-only forward maintains.
- derived fp8 and QAT caches: rewound to the pre-verify entries truncated
  to the surviving pool rows. Both transforms are per-row functions of the
  pooled rows, so later forwards re-extend them to identical values.
- fixed-decode-state branches: every restored dict value is a fresh
  object, which invalidates the branch mirrors; the fixed layout re-adopts
  from the restored values on the next decode step, exactly as it does
  after any stock-path forward.

Snapshot cost is `O(n_tokens)` recorded rows plus fixed-size window
buffers per layer, independent of context length. A snapshot is single
use: `restore_verify_state` uninstalls the recorders and consumes it, and
it must be restored (with `keep_tokens = n_tokens` for a fully accepted
block) before the next forward touches the cache.

Fail-closed rules
-----------------
Capture rejects cache objects that are not exactly `KVCache` or
`DeepseekV4Cache` (method wrappers installed on those instances are fine;
subclasses with unknown state are not). Restore validates the observed
mutations against the recorded rows: the local window and each engaged
accumulator branch must have received exactly `n_tokens` rows, and the
pool must have grown by exactly the number of completed windows. Any
mismatch raises instead of leaving a silently wrong cache.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import mlx.core as mx

from jang_tools.dsv4.mlx_model import DeepseekV4Cache
from mlx_lm.models.cache import KVCache, RotatingKVCache

_STATE_KEYS = ("compressor_state", "indexer_state")
_AUX_KEYS_BY_STATE = {
    "compressor_state": ("pooled_fp8", "pooled_fp8_rows"),
    "indexer_state": ("pooled_qat", "pooled_qat_rows"),
}
_RECORDER_FLAG = "_moespresso_dspark_rollback_recording"


def _copy(x: Optional[mx.array]) -> Optional[mx.array]:
    """Materializable copy detached from later in-place updates."""
    return None if x is None else x[...]


def _rows(x: Optional[mx.array]) -> int:
    return 0 if x is None else int(x.shape[1])


def _concat_rows(parts: List[mx.array], axis: int) -> mx.array:
    return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=axis)


class _PlainEntry:
    """Snapshot of one plain `KVCache` layer."""

    def __init__(self, cache, n_tokens: int):
        self.cache = cache
        self.offset = int(cache.offset)
        end = self.offset
        if cache.keys is not None:
            end = min(self.offset + n_tokens, int(cache.keys.shape[2]))
        self.captured_end = end
        if end > self.offset:
            self.keys = cache.keys[..., self.offset : end, :]
            self.values = cache.values[..., self.offset : end, :]
        else:
            self.keys = None
            self.values = None

    def captured_arrays(self) -> List[mx.array]:
        return [a for a in (self.keys, self.values) if a is not None]

    def uninstall(self) -> None:
        pass

    def restore(self, n_tokens: int, keep: int) -> List[mx.array]:
        cache = self.cache
        if int(cache.offset) != self.offset + n_tokens:
            raise RuntimeError(
                f"KVCache advanced to offset {int(cache.offset)}; expected "
                f"{self.offset + n_tokens} after a {n_tokens}-token verify"
            )
        cache.offset = self.offset + keep
        start = self.offset + keep
        if self.keys is not None and self.captured_end > start:
            rel = start - self.offset
            cache.keys[..., start : self.captured_end, :] = self.keys[..., rel:, :]
            cache.values[..., start : self.captured_end, :] = self.values[..., rel:, :]
            return [cache.keys, cache.values]
        return []


class _CompositeEntry:
    """Snapshot plus mutation recorders for one `DeepseekV4Cache` layer."""

    def __init__(self, cache, n_tokens: int):
        local = cache.local
        if type(local) is not RotatingKVCache:
            raise TypeError(
                "DeepseekV4Cache.local must be RotatingKVCache for bit-exact "
                f"rollback; found {type(local).__name__}"
            )
        self.cache = cache
        self.local_keys = _copy(local.keys)
        self.local_values = _copy(local.values)
        self.local_offset = int(local.offset)
        self.local_idx = int(local._idx)
        self.pre: Dict[str, dict] = {}
        for state_key in _STATE_KEYS:
            state = getattr(cache, state_key, None)
            if not isinstance(state, dict):
                raise TypeError(f"DeepseekV4Cache.{state_key} is not a dict")
            aux_key, aux_rows_key = _AUX_KEYS_BY_STATE[state_key]
            self.pre[state_key] = {
                "buffer_kv": _copy(state.get("buffer_kv")),
                "buffer_gate": _copy(state.get("buffer_gate")),
                "pool_rows": _rows(state.get("pooled")),
                # References suffice for the derived caches: array updates
                # rebind handles instead of mutating referenced nodes, so
                # pre-verify contents stay reachable without a copy.
                "aux": state.get(aux_key),
                "aux_rows": int(state.get(aux_rows_key, 0) or 0),
            }
        self.local_records: List[Tuple[mx.array, mx.array]] = []
        self.accum_records: Dict[str, dict] = {}
        self._install()

    def captured_arrays(self) -> List[mx.array]:
        arrays = [a for a in (self.local_keys, self.local_values) if a is not None]
        for pre in self.pre.values():
            arrays += [a for a in (pre["buffer_kv"], pre["buffer_gate"]) if a is not None]
        return arrays

    def _install(self) -> None:
        cache = self.cache
        local = cache.local
        if getattr(cache, _RECORDER_FLAG, False):
            raise RuntimeError(
                "a verify snapshot is already recording on this cache; "
                "restore it before capturing again"
            )
        self._previous = {
            "update_and_fetch": local.__dict__.get("update_and_fetch"),
            "accumulate_windows": cache.__dict__.get("accumulate_windows"),
            "accumulate_overlap_windows": cache.__dict__.get(
                "accumulate_overlap_windows"
            ),
        }

        original_update = local.update_and_fetch
        local_records = self.local_records

        def recording_update(keys, values):
            local_records.append((keys, values))
            return original_update(keys, values)

        accum_records = self.accum_records

        def _record(state_key, ratio, overlap, kv, gate):
            slot = accum_records.get(state_key)
            if slot is None:
                slot = {"kv": [], "gate": [], "ratio": int(ratio), "overlap": overlap}
                accum_records[state_key] = slot
            elif slot["ratio"] != int(ratio) or slot["overlap"] != overlap:
                raise RuntimeError(
                    f"inconsistent accumulate calls for {state_key} within one "
                    "verify forward"
                )
            slot["kv"].append(kv)
            slot["gate"].append(gate)

        original_windows = cache.accumulate_windows

        def recording_windows(kv, gate, state_key, ratio, start_pos):
            _record(state_key, ratio, False, kv, gate)
            return original_windows(kv, gate, state_key, ratio, start_pos)

        original_overlap = cache.accumulate_overlap_windows

        def recording_overlap(kv, gate, state_key, ratio, start_pos, head_dim):
            _record(state_key, ratio, True, kv, gate)
            return original_overlap(kv, gate, state_key, ratio, start_pos, head_dim)

        object.__setattr__(local, "update_and_fetch", recording_update)
        object.__setattr__(cache, "accumulate_windows", recording_windows)
        object.__setattr__(cache, "accumulate_overlap_windows", recording_overlap)
        object.__setattr__(cache, _RECORDER_FLAG, True)

    def uninstall(self) -> None:
        cache = self.cache
        if not getattr(cache, _RECORDER_FLAG, False):
            return
        targets = (
            ("update_and_fetch", cache.local),
            ("accumulate_windows", cache),
            ("accumulate_overlap_windows", cache),
        )
        for name, target in targets:
            previous = self._previous[name]
            if previous is None:
                object.__delattr__(target, name)
            else:
                object.__setattr__(target, name, previous)
        object.__setattr__(cache, _RECORDER_FLAG, False)

    def _restore_local(self, n_tokens: int, keep: int) -> List[mx.array]:
        local = self.cache.local
        if int(local.offset) != self.local_offset + n_tokens:
            raise RuntimeError(
                f"local window advanced to offset {int(local.offset)}; expected "
                f"{self.local_offset + n_tokens} after a {n_tokens}-token verify"
            )
        recorded = sum(int(k.shape[2]) for k, _ in self.local_records)
        if recorded != n_tokens:
            raise RuntimeError(
                f"local window recorded {recorded} rows for a "
                f"{n_tokens}-token verify"
            )
        keys = _concat_rows([k for k, _ in self.local_records], axis=2)
        values = _concat_rows([v for _, v in self.local_records], axis=2)
        local.keys = self.local_keys
        local.values = self.local_values
        local.offset = self.local_offset
        local._idx = self.local_idx
        for i in range(keep):
            local._update_in_place(keys[..., i : i + 1, :], values[..., i : i + 1, :])
        if local.keys is None:
            return []
        return [local.keys, local.values]

    def _restore_branch(self, state_key: str, n_tokens: int, keep: int) -> List[mx.array]:
        slot = self.accum_records.get(state_key)
        if slot is None:
            # The forward never accumulated into this branch, so nothing in
            # it was mutated and a kept-only forward would not have touched
            # it either.
            return []
        state = getattr(self.cache, state_key)
        pre = self.pre[state_key]
        ratio = slot["ratio"]
        overlap = slot["overlap"]
        rec_kv = _concat_rows(slot["kv"], axis=1)
        rec_gate = _concat_rows(slot["gate"], axis=1)
        if _rows(rec_kv) != n_tokens or _rows(rec_gate) != n_tokens:
            raise RuntimeError(
                f"{state_key} recorded {_rows(rec_kv)} rows for a "
                f"{n_tokens}-token verify"
            )

        pre_kv = pre["buffer_kv"]
        pre_gate = pre["buffer_gate"]
        pre_len = _rows(pre_kv)
        pre_prev = ratio if (overlap and pre_len >= ratio) else 0
        partial0 = pre_len - pre_prev
        appended_total = (partial0 + n_tokens) // ratio
        post_pooled = state.get("pooled")
        post_rows = _rows(post_pooled)
        if post_rows != pre["pool_rows"] + appended_total:
            raise RuntimeError(
                f"{state_key} pool grew from {pre['pool_rows']} to {post_rows} "
                f"rows; expected {appended_total} appended windows"
            )

        appended_keep = (partial0 + keep) // ratio
        target_rows = pre["pool_rows"] + appended_keep
        new_partial = (partial0 + keep) % ratio
        prev_after = overlap and (pre_prev > 0 or appended_keep > 0)
        target_len = new_partial + (ratio if prev_after else 0)

        kv_parts = [rec_kv[:, :keep]]
        gate_parts = [rec_gate[:, :keep]]
        if pre_len:
            kv_parts.insert(0, pre_kv)
            gate_parts.insert(0, pre_gate)
        combined_kv = _concat_rows(kv_parts, axis=1)
        combined_gate = _concat_rows(gate_parts, axis=1)
        tail = int(combined_kv.shape[1]) - target_len
        state["buffer_kv"] = combined_kv[:, tail:]
        state["buffer_gate"] = combined_gate[:, tail:]
        state["pooled"] = post_pooled[:, :target_rows] if target_rows else None

        aux_key, aux_rows_key = _AUX_KEYS_BY_STATE[state_key]
        aux = pre["aux"]
        aux_rows = min(pre["aux_rows"], target_rows)
        if aux is None or aux_rows <= 0:
            state.pop(aux_key, None)
            state.pop(aux_rows_key, None)
        else:
            state[aux_key] = aux[:, :aux_rows]
            state[aux_rows_key] = aux_rows

        touched = [state["buffer_kv"], state["buffer_gate"]]
        if state["pooled"] is not None:
            touched.append(state["pooled"])
        return touched

    def restore(self, n_tokens: int, keep: int) -> List[mx.array]:
        touched = self._restore_local(n_tokens, keep)
        for state_key in _STATE_KEYS:
            touched += self._restore_branch(state_key, n_tokens, keep)
        return touched


class VerifySnapshot:
    """Single-use pre-verify snapshot for one layer-cache list."""

    def __init__(self, n_tokens: int, entries: List[object]):
        self.n_tokens = n_tokens
        self.entries = entries
        self.consumed = False


def capture_verify_state(cache: Sequence, n_tokens: int) -> VerifySnapshot:
    """Snapshot every layer cache before a verify forward of `n_tokens`.

    Installs transient row recorders on the composite caches; the snapshot
    must be consumed by `restore_verify_state` after the forward.
    """
    n_tokens = int(n_tokens)
    if n_tokens < 1:
        raise ValueError(f"n_tokens must be at least 1, got {n_tokens}")
    entries: List[object] = []
    captured: List[mx.array] = []
    try:
        for layer_cache in cache:
            if type(layer_cache) is DeepseekV4Cache:
                entry = _CompositeEntry(layer_cache, n_tokens)
            elif type(layer_cache) is KVCache:
                entry = _PlainEntry(layer_cache, n_tokens)
            else:
                raise TypeError(
                    "bit-exact verify rollback supports KVCache and "
                    f"DeepseekV4Cache layers; found {type(layer_cache).__name__}"
                )
            entries.append(entry)
            captured += entry.captured_arrays()
    except BaseException:
        for entry in entries:
            entry.uninstall()
        raise
    if captured:
        mx.eval(*captured)
    return VerifySnapshot(n_tokens, entries)


def restore_verify_state(cache: Sequence, snapshot: VerifySnapshot, keep_tokens: int) -> None:
    """Rewind the caches to the first `keep_tokens` positions of the block.

    `keep_tokens` may be 0 (full rollback to the pre-verify state) up to
    `snapshot.n_tokens` (keep the whole block; the recorders are removed
    and the ring layout is normalized, nothing else changes).
    """
    keep_tokens = int(keep_tokens)
    if snapshot.consumed:
        raise RuntimeError("verify snapshot already consumed")
    if not 0 <= keep_tokens <= snapshot.n_tokens:
        raise ValueError(
            f"keep_tokens must be in [0, {snapshot.n_tokens}], got {keep_tokens}"
        )
    caches = list(cache)
    if len(caches) != len(snapshot.entries):
        raise ValueError(
            f"snapshot covers {len(snapshot.entries)} layer caches; "
            f"got {len(caches)}"
        )
    for entry, layer_cache in zip(snapshot.entries, caches):
        if entry.cache is not layer_cache:
            raise ValueError("snapshot was captured from a different cache list")
    snapshot.consumed = True
    for entry in snapshot.entries:
        entry.uninstall()
    touched: List[mx.array] = []
    for entry in snapshot.entries:
        touched += entry.restore(snapshot.n_tokens, keep_tokens)
    if touched:
        mx.eval(*touched)
