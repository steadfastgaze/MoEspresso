"""Durable tensor representation of committed Qwen composite prompt state."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import mlx.core as mx

from moespresso.runtime.qwen4.gdn import Qwen4GDNState
from moespresso.runtime.qwen4.kvarn_snapshot import restore_kvarn_state, snapshot_kvarn_state
from moespresso.runtime.qwen4.model import Qwen4CompositeState, Qwen4LayerState, Qwen4TextModelShell
from moespresso.runtime.qwen4.ple import Qwen4PLEState
from moespresso.runtime.qwen4.qsa_kvarn import Qwen4MutableKVarNQSAStateBackend


QWEN4_SNAPSHOT_SCHEMA = "qwen4-composite-kvarn4-snapshot-v1"
_META_FIELDS = frozenset({"schema", "cache_identity", "frontier", "revision", "layers"})


def snapshot_composite_state(
    model: Qwen4TextModelShell, state: Qwen4CompositeState,
) -> tuple[tuple, dict[str, Any]]:
    """Capture one complete committed text prefix with independent tensor ownership."""
    model.validate_state(state)
    if state.batch_size != 1 or state.frontier <= 0 or not bool(mx.all(state.valid_history).item()):
        raise ValueError("Qwen disk snapshots require a nonempty all-valid text prefix")
    trees: list = [(mx.array(state.valid_history), mx.array(state.position_history))]
    layers = []
    for layer in state.layers:
        if layer.mixer_kind == "gdn" and isinstance(layer.mixer_state, Qwen4GDNState):
            mixer = layer.mixer_state
            arrays = (mx.array(mixer.conv_state), mx.array(mixer.recurrent_state))
            meta = {"schema": mixer.schema, "recurrent_layout": mixer.recurrent_layout}
        elif layer.mixer_kind == "qsa":
            arrays, meta = snapshot_kvarn_state(layer.mixer_state)
        else:
            raise ValueError("Qwen disk snapshot has an unsupported mixer")
        ple = layer.ple_state
        ple_arrays = None if ple is None else (mx.array(ple.token_context), mx.array(ple.conv_state))
        trees.append((arrays, ple_arrays))
        layers.append({"kind": layer.mixer_kind, "mixer": meta, "ple": ple is not None})
    mx.eval(trees)
    return tuple(trees), {
        "schema": QWEN4_SNAPSHOT_SCHEMA,
        "cache_identity": state.cache_identity,
        "frontier": state.frontier,
        "revision": state.revision,
        "layers": layers,
    }


def _sequence(value, length: int, name: str):
    if not isinstance(value, (tuple, list)) or len(value) != length:
        raise ValueError(f"Qwen snapshot {name} has incompatible length")
    return value


def _arrays(value, length: int, name: str):
    result = _sequence(value, length, name)
    if not all(isinstance(array, mx.array) for array in result):
        raise ValueError(f"Qwen snapshot {name} contains a non-array")
    return result


def restore_composite_state(
    model: Qwen4TextModelShell, trees, metadata: Mapping[str, Any], *, expected_frontier: int,
) -> Qwen4CompositeState:
    """Build and validate complete state before exposing it to a coordinator.

    The caller supplies the exact token-prefix frontier selected by the disk
    index. Packed QSA records restore without requantization. New storage owners
    and frontier markers replace all process-local rollback identities.
    """
    if not isinstance(metadata, Mapping) or set(metadata) != _META_FIELDS:
        raise ValueError("Qwen snapshot metadata fields are incompatible")
    if metadata["schema"] != QWEN4_SNAPSHOT_SCHEMA:
        raise ValueError("Qwen snapshot schema is incompatible")
    if metadata["cache_identity"] != model.cache_identity:
        raise ValueError("Qwen snapshot cache identity does not match the model")
    frontier, revision = metadata["frontier"], metadata["revision"]
    if (
        isinstance(frontier, bool) or not isinstance(frontier, int) or frontier <= 0
        or isinstance(expected_frontier, bool) or not isinstance(expected_frontier, int)
        or frontier != expected_frontier
        or isinstance(revision, bool) or not isinstance(revision, int) or revision < 0
    ):
        raise ValueError("Qwen snapshot is off the selected token frontier")
    trees = _sequence(trees, len(model.layers) + 1, "tensor tree")
    layer_meta = _sequence(metadata["layers"], len(model.layers), "layer metadata")
    valid, positions = _arrays(trees[0], 2, "position history")
    if valid.shape != (1, frontier) or valid.dtype != mx.bool_:
        raise ValueError("Qwen snapshot validity mask is incompatible")
    if positions.shape != (3, 1, frontier) or positions.dtype not in (
        mx.int32, mx.int64, mx.uint32, mx.uint64,
    ):
        raise ValueError("Qwen snapshot position history is incompatible")
    if not bool(mx.all(valid).item()):
        raise ValueError("Qwen disk snapshots require an all-valid text prefix")
    layers = []
    finite = []
    for module, tree, meta in zip(model.layers, trees[1:], layer_meta, strict=True):
        if not isinstance(meta, Mapping) or set(meta) != {"kind", "mixer", "ple"}:
            raise ValueError("Qwen snapshot layer metadata is incompatible")
        if meta["kind"] != module.mixer_kind or meta["ple"] is not (module.ple is not None):
            raise ValueError("Qwen snapshot layer layout does not match the model")
        mixer_tree, ple_tree = _sequence(tree, 2, "layer state")
        mixer_meta = meta["mixer"]
        if module.mixer_kind == "gdn":
            if not isinstance(mixer_meta, Mapping) or set(mixer_meta) != {"schema", "recurrent_layout"}:
                raise ValueError("Qwen snapshot GDN metadata is incompatible")
            conv, recurrent = _arrays(mixer_tree, 2, "GDN state")
            mixer_state = Qwen4GDNState(
                conv_state=mx.array(conv), recurrent_state=mx.array(recurrent), offset=frontier,
                schema=mixer_meta["schema"], recurrent_layout=mixer_meta["recurrent_layout"],
            )
            finite.extend((mx.all(mx.isfinite(conv)), mx.all(mx.isfinite(recurrent))))
        elif module.mixer_kind == "qsa":
            backend = getattr(module.mixer, "state_backend", None)
            if not isinstance(backend, Qwen4MutableKVarNQSAStateBackend):
                raise ValueError("Qwen snapshot requires the mutable KVarN4 backend")
            if not isinstance(mixer_meta, Mapping) or mixer_meta.get("frontier") != frontier:
                raise ValueError("Qwen snapshot QSA layer is off the common frontier")
            mixer_state = restore_kvarn_state(
                mixer_tree, mixer_meta, max_context_tokens=backend.max_context_tokens,
            )
        else:
            raise ValueError("Qwen snapshot mixer kind is unsupported")
        ple_state = None
        if module.ple is not None:
            tokens, conv = _arrays(ple_tree, 2, "PLE state")
            ple_state = Qwen4PLEState(mx.array(tokens), mx.array(conv), frontier)
            finite.append(mx.all(mx.isfinite(conv)))
        else:
            if ple_tree is not None:
                raise ValueError("Qwen snapshot contains unexpected PLE state")
        layers.append(Qwen4LayerState(module.mixer_kind, mixer_state, frontier, ple_state))
    state = Qwen4CompositeState(
        cache_identity=model.cache_identity, revision=revision, frontier=frontier,
        batch_size=1, valid_history=mx.array(valid), position_history=mx.array(positions),
        layers=tuple(layers),
    )
    model.validate_state(state)
    mx.eval(*finite)
    if not all(bool(value.item()) for value in finite):
        raise ValueError("Qwen snapshot contains non-finite recurrent or PLE state")
    return state
