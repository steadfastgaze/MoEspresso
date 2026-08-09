"""Restart coverage for paired DSpark disk-KV checkpoints.

The test uses the production target and attachment safetensors codecs.  It
keeps the model synthetic so the evidence isolates persistence, reconstruction,
and the model-specific continuation bridge without running a full decode.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

import mlx.core as mx  # noqa: E402
from mlx_lm.models.cache import KVCache  # noqa: E402

from moespresso.runtime.deepseek_v4.spec_decode import (  # noqa: E402
    CapsuleScalar,
    DrafterStateCapsule,
)
from moespresso.runtime.deepseek_v4.spec_disk_kv import (  # noqa: E402
    build_spec_attachment_identity,
    restore_spec_attachment,
)
from moespresso.runtime.deepseek_v4.spec_serve import (  # noqa: E402
    ServedDrafter,
    spec_cache_producer_rail,
)
from moespresso.runtime.disk_kv import (  # noqa: E402
    ATTACHMENT_STATUS_HIT,
    ATTACHMENT_STATUS_INVALID,
    DiskCheckpointStore,
    DiskKVAttachmentEnvelope,
    build_cache_scope,
    default_cache_registry,
    load_attachment_payload,
    load_prompt_cache_payload,
    save_attachment_payload,
    save_prompt_cache_payload,
)


_MODEL_KEY = ("pkg", "render", "raw", 64, 0, "mlx_prompt_cache")
_CAPSULE_KIND = "deepseek_v4_dspark_window_state"


@dataclass(frozen=True)
class _DSparkState:
    frontier: int
    metadata: tuple[tuple[str, CapsuleScalar], ...]
    tensors: tuple[mx.array, ...]


class _CapsuleDrafter:
    state_capsule_kind = _CAPSULE_KIND
    state_capsule_schema_major = 1
    state_capsule_schema_minor = 0

    def __init__(self) -> None:
        self.last_imported: _DSparkState | None = None

    def state_frontier(self, state: _DSparkState) -> int:
        return state.frontier

    def state_nbytes(self, state: _DSparkState) -> int:
        return sum(tensor.nbytes for tensor in state.tensors)

    def export_state(self, state: _DSparkState) -> DrafterStateCapsule:
        return DrafterStateCapsule(
            kind=self.state_capsule_kind,
            schema_major=self.state_capsule_schema_major,
            schema_minor=self.state_capsule_schema_minor,
            frontier=state.frontier,
            metadata=state.metadata,
            tensors=state.tensors,
        )

    def import_state(self, capsule: DrafterStateCapsule) -> _DSparkState:
        if capsule.kind != self.state_capsule_kind:
            raise ValueError("capsule kind mismatch")
        if capsule.schema_major != self.state_capsule_schema_major:
            raise ValueError("capsule major version mismatch")
        if capsule.schema_minor != self.state_capsule_schema_minor:
            raise ValueError("capsule minor version mismatch")
        state = _DSparkState(
            frontier=capsule.frontier,
            metadata=capsule.metadata,
            tensors=capsule.tensors,
        )
        self.last_imported = state
        return state


def _served(drafter: _CapsuleDrafter) -> ServedDrafter:
    return ServedDrafter(
        family="dspark",
        sidecar_dir=Path("synthetic/dspark"),
        drafter=drafter,
        tap=None,
        artifact_id="art:synthetic-dspark",
    )


def _target_cache(frontier: int) -> KVCache:
    cache = KVCache()
    shape = (1, 2, frontier, 8)
    values = mx.arange(int(np.prod(shape)), dtype=mx.float32).reshape(shape)
    keys = (values / 17.0).astype(mx.float16)
    values = (-values / 29.0).astype(mx.float16)
    cache.update_and_fetch(keys, values)
    mx.eval(cache.state)
    return cache


def _empty_target_cache() -> KVCache:
    return KVCache()


def _drafter_state(frontier: int) -> _DSparkState:
    signed = mx.array(
        np.array([0.0, -0.0, 1.25, -2.5], dtype=np.float32))
    ring = mx.arange(96, dtype=mx.float32).reshape(2, 6, 8).astype(mx.float16)
    valid = mx.array([0, 7, frontier - 1], dtype=mx.int32)
    empty = mx.zeros((0, 8), dtype=mx.float16)
    mx.eval(signed, ring, valid, empty)
    return _DSparkState(
        frontier=frontier,
        metadata=(
            ("stages", 3),
            ("window", 6),
            ("wrapped", True),
            ("label", "restart-fixture"),
        ),
        tensors=(signed, ring, valid, empty),
    )


def _array_signature(value: mx.array) -> tuple[tuple[int, ...], str, bytes]:
    array = np.asarray(value)
    return tuple(array.shape), str(array.dtype), array.tobytes(order="C")


def _assert_tensors_bit_identical(
    actual: tuple[mx.array, ...] | list[mx.array],
    expected: tuple[mx.array, ...] | list[mx.array],
) -> None:
    assert len(actual) == len(expected)
    assert [_array_signature(value) for value in actual] == [
        _array_signature(value) for value in expected
    ]


def test_dspark_target_and_capsule_survive_restart_and_corrupt_isolation(tmp_path):
    frontier = 256
    tokens = list(range(frontier))
    suffix = [997, 998]
    source_cache = _target_cache(frontier)
    source_state = _drafter_state(frontier)
    source_drafter = _CapsuleDrafter()
    source_served = _served(source_drafter)
    producer_rail = spec_cache_producer_rail(
        source_served, schedule="fixed:3")
    assert producer_rail is not None

    class_names = (type(source_cache).__name__,)
    scope = build_cache_scope(_MODEL_KEY, class_names)
    write_order: list[str] = []

    def save_target(*args, **kwargs):
        write_order.append("target")
        return save_prompt_cache_payload(*args, **kwargs)

    def save_companion(*args, **kwargs):
        write_order.append("attachment")
        return save_attachment_payload(*args, **kwargs)

    store = DiskCheckpointStore(
        tmp_path,
        stride=frontier,
        save_payload_fn=save_target,
        save_attachment_payload_fn=save_companion,
        log_fn=lambda line: None,
    )
    target = store.write_manual_checkpoint(
        scope,
        tokens,
        cache_state_trees=[source_cache.state],
        meta_state_trees=[source_cache.meta_state],
        cache_class_names=class_names,
        now=11,
    )
    assert target is not None
    assert write_order == ["target"]
    identity = build_spec_attachment_identity(
        target, source_served, producer_rail)
    source_capsule = source_drafter.export_state(source_state)
    attachment = store.write_attachment(
        target,
        DiskKVAttachmentEnvelope.from_capsule(identity, source_capsule),
        now=12,
    )
    assert attachment is not None
    assert write_order == ["target", "attachment"]
    assert store.find_exact(scope, tokens) is not None
    assert store.has_attachment(target, identity)
    target_payload = tmp_path / target.payload_path
    attachment_payload = tmp_path / attachment.payload_path
    assert target_payload.is_file()
    assert attachment_payload.is_file()
    store.close()

    restore_order: list[str] = []

    def load_target(*args, **kwargs):
        restore_order.append("target")
        return load_prompt_cache_payload(*args, **kwargs)

    def load_companion(*args, **kwargs):
        restore_order.append("attachment")
        return load_attachment_payload(*args, **kwargs)

    resumed_drafter = _CapsuleDrafter()
    resumed_served = _served(resumed_drafter)
    resumed_rail = spec_cache_producer_rail(
        resumed_served, schedule="fixed:3")
    assert resumed_rail == producer_rail
    reopened = DiskCheckpointStore(
        tmp_path,
        stride=frontier,
        load_payload_fn=load_target,
        load_attachment_payload_fn=load_companion,
        log_fn=lambda line: None,
    )
    hit = reopened.restore(
        scope,
        tokens + suffix,
        make_cache_fn=lambda: [_empty_target_cache()],
        registry=default_cache_registry(),
    )
    assert hit is not None
    assert restore_order == ["target"]
    assert hit.cached_tokens == frontier
    assert hit.suffix_tokens == suffix
    _assert_tensors_bit_identical(hit.prompt_cache[0].state, source_cache.state)

    restored = restore_spec_attachment(
        reopened, hit, resumed_served, resumed_rail)
    assert restore_order == ["target", "attachment"]
    assert restored.status == ATTACHMENT_STATUS_HIT
    assert restored.continuation is not None
    continuation = restored.continuation
    assert continuation.target_cache is hit.prompt_cache
    assert continuation.prefix_offset == frontier
    assert continuation.companion.frontier == frontier
    assert continuation.companion.producer_rail == producer_rail
    assert continuation.companion.capsule.kind == _CAPSULE_KIND
    assert continuation.companion.capsule.frontier == frontier
    assert continuation.companion.capsule.metadata == source_capsule.metadata
    _assert_tensors_bit_identical(
        continuation.companion.capsule.tensors, source_capsule.tensors)
    assert resumed_drafter.last_imported is not None
    assert resumed_drafter.last_imported.frontier == frontier
    assert resumed_drafter.last_imported.metadata == source_state.metadata
    _assert_tensors_bit_identical(
        resumed_drafter.last_imported.tensors, source_state.tensors)
    reopened.close()

    with open(attachment_payload, "r+b") as handle:
        handle.truncate(max(1, attachment_payload.stat().st_size // 2))

    fallback_drafter = _CapsuleDrafter()
    fallback_served = _served(fallback_drafter)
    fallback_rail = spec_cache_producer_rail(
        fallback_served, schedule="fixed:3")
    assert fallback_rail == producer_rail
    after_corruption = DiskCheckpointStore(
        tmp_path,
        stride=frontier,
        log_fn=lambda line: None,
    )
    target_only_hit = after_corruption.restore(
        scope,
        tokens + suffix,
        make_cache_fn=lambda: [_empty_target_cache()],
        registry=default_cache_registry(),
    )
    assert target_only_hit is not None
    _assert_tensors_bit_identical(
        target_only_hit.prompt_cache[0].state, source_cache.state)
    invalid = restore_spec_attachment(
        after_corruption, target_only_hit, fallback_served, fallback_rail)
    assert invalid.status == ATTACHMENT_STATUS_INVALID
    assert invalid.continuation is None
    assert fallback_drafter.last_imported is None
    assert target_payload.is_file()
    assert after_corruption.find_exact(scope, tokens) is not None
    assert list((tmp_path / "attachments" / "quarantine").glob("*.safetensors"))
    after_corruption.close()
