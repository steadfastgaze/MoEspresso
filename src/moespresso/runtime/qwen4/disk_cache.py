"""Qwen composite prompt checkpoints over the shared memory and disk stores."""

from __future__ import annotations

from collections.abc import Mapping
import time

import mlx.core as mx

from moespresso.runtime.disk_kv import (
    DiskKVInvalidPayload,
    FrontierTracker,
    build_cache_scope,
    plan_prefill_chunks,
)
from moespresso.runtime.qwen4.cache_snapshot import restore_composite_state, snapshot_composite_state


QWEN4_PROMPT_CACHE_SCHEMA = "qwen4-prompt-kvarn4-logits-v2"


def qwen4_cache_scope(model, rendering_identity: str) -> dict:
    """Bind checkpoints to the package, rendering, routing and tensor contracts."""
    return build_cache_scope(
        (model.cache_identity, rendering_identity, "qwen_kvarn_k4v4", 128, 0,
         QWEN4_PROMPT_CACHE_SCHEMA),
        (Qwen4PromptCache.__name__,),
    )


def _leaves(tree):
    if isinstance(tree, mx.array):
        yield tree
    elif isinstance(tree, (tuple, list)):
        for child in tree:
            yield from _leaves(child)


class Qwen4PromptCache:
    """Compact immutable tensor payload; live mutable ownership is rebuilt on use."""

    def __init__(self):
        self.state = None
        self.meta_state = None
        self._restored = None

    @classmethod
    def capture(cls, model, state, logits):
        trees, metadata = snapshot_composite_state(model, state)
        last_logits = mx.array(logits[:, -1:, :])
        mx.eval(last_logits)
        result = cls()
        result.state = (trees, last_logits)
        result.meta_state = {"schema": QWEN4_PROMPT_CACHE_SCHEMA, "composite": metadata}
        return result

    @property
    def offset(self) -> int:
        return int(self.meta_state["composite"]["frontier"])

    @property
    def nbytes(self) -> int:
        return sum(array.nbytes for array in _leaves(self.state))

    @staticmethod
    def is_trimmable() -> bool:
        return False

    def restore(self, model, expected_frontier: int):
        """Validate all durable fields before publishing a live cache owner."""
        self._restored = None
        try:
            meta = self.meta_state
            if not isinstance(meta, Mapping) or set(meta) != {"schema", "composite"}:
                raise ValueError("Qwen prompt checkpoint metadata is incompatible")
            if meta["schema"] != QWEN4_PROMPT_CACHE_SCHEMA:
                raise ValueError("Qwen prompt checkpoint schema is incompatible")
            if not isinstance(self.state, (tuple, list)) or len(self.state) != 2:
                raise ValueError("Qwen prompt checkpoint tensor tree is incompatible")
            trees, logits = self.state
            vocabulary = int(model.lm_head.weight.shape[0])
            if (
                not isinstance(logits, mx.array) or logits.shape != (1, 1, vocabulary)
                or logits.dtype not in (mx.float32, mx.bfloat16, mx.float16)
                or not bool(mx.all(mx.isfinite(logits)).item())
            ):
                raise ValueError("Qwen prompt checkpoint logits are incompatible")
            state = restore_composite_state(
                model, trees, meta["composite"], expected_frontier=expected_frontier,
            )
        except (ValueError, TypeError, KeyError, IndexError) as error:
            raise DiskKVInvalidPayload(f"invalid Qwen prompt checkpoint: {error}") from error
        self._restored = (state, logits)
        return self._restored

    def take_restored(self):
        if self._restored is None:
            raise RuntimeError("Qwen prompt checkpoint has not been validated")
        restored, self._restored = self._restored, None
        self.state = None
        self.meta_state = None
        return restored


class Qwen4CheckpointWriter:
    """Capture aligned committed prefill state through the existing disk writer."""

    def __init__(self, model, store, scope, full_tokens, cached_tokens, session_cache_key=None):
        self.model = model
        self.store = store
        self.session_cache_key = session_cache_key
        self.tracker = FrontierTracker(
            stride=store.stride, restored_prefix=cached_tokens, full_tokens=full_tokens,
            scope=scope, already_written=store.has_entry, write_depth=store.write_depth_tokens,
        )
        self.written = []
        self.write_seconds = []
        self.disabled = False

    def prefill_plan(self, step):
        boundaries = []
        position = self.tracker.restored_prefix
        while True:
            frontier = self.tracker.next_frontier_above(position)
            if frontier is None or frontier > len(self.tracker.full_tokens):
                break
            boundaries.append(frontier)
            position = frontier
        return plan_prefill_chunks(start=self.tracker.restored_prefix, boundaries=boundaries, step=step)

    def capture(self, state, logits):
        if self.disabled or self.store.writes_disabled:
            return
        for frontier in self.tracker.crossings_up_to(state.frontier):
            if frontier != state.frontier:
                continue
            started = time.perf_counter()
            try:
                cache = Qwen4PromptCache.capture(self.model, state, logits)
                if cache.offset != frontier:
                    raise ValueError("Qwen checkpoint state is off the proposed frontier")
                entry = self.store.write_checkpoint(
                    self.tracker.scope, self.tracker.frontier_tokens(frontier),
                    cache_state_trees=[cache.state], meta_state_trees=[cache.meta_state],
                    cache_class_names=(Qwen4PromptCache.__name__,), reason="aligned_frontier",
                    session_cache_key=self.session_cache_key, now=int(time.time()),
                )
                if entry is not None:
                    self.written.append(entry)
                    self.write_seconds.append(time.perf_counter() - started)
            except Exception as error:  # noqa: BLE001 - persistence must not fail generation
                self.disabled = True
                self.store._log(
                    f"[disk_kv] write failed token_count={frontier}; "
                    f"disabling checkpoint writes for this request: {error!r}"
                )


def validate_qwen4_checkpoint(caches, entry, model):
    """Validate the model-specific payload before the store records a restore."""
    if len(caches) != 1 or not isinstance(caches[0], Qwen4PromptCache):
        raise DiskKVInvalidPayload("Qwen disk checkpoint requires one composite cache")
    if entry.scope.get("cache_payload_kind") != QWEN4_PROMPT_CACHE_SCHEMA:
        raise DiskKVInvalidPayload("Qwen disk checkpoint payload kind is incompatible")
    caches[0].restore(model, entry.token_count)
