"""Fail-closed package hydration for the manifest-built Qwen4 graph."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from moespresso.package.iqk_format import (
    IQK_LAYOUT_IQK_RELAYOUT,
    IQK_LAYOUT_QWEN4_STREAM_MAJOR_V1,
)


class Qwen4PackageLoadError(RuntimeError):
    """Raised when declared package bytes cannot hydrate the Qwen4 graph."""


_RAW_FORMATS = frozenset({"raw_dtype_passthrough", "fp16"})
_SUPPORTED_DIRECT_FORMATS = _RAW_FORMATS | {"iqk", "kquant"}


def _fail(message: str) -> Qwen4PackageLoadError:
    return Qwen4PackageLoadError(message)


def _declared_shard(package_dir: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise _fail("tensor shard must be a non-empty string")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        "\\" in value
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise _fail(f"tensor shard {value!r} escapes the package root")
    try:
        root = package_dir.resolve()
        path = (root / value).resolve()
        path.relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise _fail(f"tensor shard {value!r} escapes the package root") from exc
    if not path.is_file():
        raise _fail(f"declared tensor shard is missing: {value}")
    return path


def _direct_keys(entry: Mapping[str, object]) -> tuple[tuple[str, str], ...]:
    prefix = entry.get("key_prefix")
    target = entry.get("module_weight_key")
    if not isinstance(prefix, str) or not prefix:
        raise _fail(f"{entry.get('source_name')}: key_prefix must be a non-empty string")
    if not isinstance(target, str) or not target:
        raise _fail(f"{entry.get('source_name')}: module_weight_key must be declared")
    fmt = entry.get("format")
    if fmt in _RAW_FORMATS:
        return ((prefix, target),)
    target_prefix = target.removesuffix(".weight")
    if fmt == "kquant":
        suffixes = ("weight", "scales")
    elif fmt == "iqk":
        suffixes = ("weight",)
    else:  # pragma: no cover - caller restricts formats
        raise _fail(f"unsupported direct format {fmt!r}")
    return tuple((f"{prefix}.{suffix}", f"{target_prefix}.{suffix}") for suffix in suffixes)


def _prepare_array(
    entry: Mapping[str, object],
    source_key: str,
    array: mx.array,
    expected: mx.array,
) -> mx.array:
    source_name = entry.get("source_name")
    fmt = entry.get("format")
    value = array
    if (
        fmt in _RAW_FORMATS
        and isinstance(source_name, str)
        and source_name.endswith(".conv1d.weight")
        and value.ndim == 3
        and tuple(value.shape) != tuple(expected.shape)
    ):
        value = mx.swapaxes(value, 1, 2)
    if tuple(value.shape) != tuple(expected.shape):
        raise _fail(
            f"{source_name}: package key {source_key} has shape {tuple(value.shape)}; "
            f"destination expects {tuple(expected.shape)}"
        )
    if fmt == "raw_dtype_passthrough" and value.dtype != mx.bfloat16:
        raise _fail(f"{source_name}: released raw Qwen4 tensors must be BF16")
    if fmt == "fp16" and value.dtype != mx.float16:
        raise _fail(f"{source_name}: fp16 package tensor is not float16")
    if fmt in {"iqk", "kquant"} and value.dtype != expected.dtype:
        raise _fail(
            f"{source_name}: package key {source_key} has dtype {value.dtype}; "
            f"destination expects {expected.dtype}"
        )
    return value


def hydrate_qwen4_package_weights(
    model: nn.Module,
    manifest: Mapping[str, object],
    package_dir: str | Path,
    *,
    shard_loader: Callable[[str], Mapping[str, mx.array]] = mx.load,
) -> int:
    """Load direct Qwen4 weights one package shard at a time.

    Expert bundle rows and PLE provider rows are not graph parameters and are
    deliberately skipped. The caller installs their providers while building
    the graph.
    """
    tensors = manifest.get("tensors")
    if not isinstance(tensors, list):
        raise _fail("manifest tensors must be an array")
    root = Path(package_dir)
    parameters = dict(tree_flatten(model.parameters()))
    destinations = set()
    by_shard: dict[Path, list[tuple[Mapping[str, object], str, str]]] = defaultdict(list)

    for index, entry in enumerate(tensors):
        if not isinstance(entry, Mapping):
            raise _fail(f"manifest tensor {index} must be an object")
        if entry.get("kind") == "expert":
            continue
        fmt = entry.get("format")
        if fmt not in _SUPPORTED_DIRECT_FORMATS:
            raise _fail(f"{entry.get('source_name')}: unsupported Qwen4 direct format {fmt!r}")
        shard = _declared_shard(root, entry.get("shard"))
        for source_key, target_key in _direct_keys(entry):
            if target_key in destinations:
                raise _fail(f"duplicate Qwen4 parameter destination: {target_key}")
            if target_key not in parameters:
                raise _fail(f"manifest parameter destination does not exist: {target_key}")
            destinations.add(target_key)
            by_shard[shard].append((entry, source_key, target_key))

    loaded_count = 0
    for shard in sorted(by_shard, key=str):
        payload = shard_loader(str(shard))
        if not isinstance(payload, Mapping):
            raise _fail(f"shard loader returned no tensor mapping for {shard.name}")
        updates = []
        for entry, source_key, target_key in by_shard[shard]:
            value = payload.get(source_key)
            if not isinstance(value, mx.array):
                raise _fail(f"{shard.name} is missing declared tensor key {source_key}")
            updates.append(
                (
                    target_key,
                    _prepare_array(entry, source_key, value, parameters[target_key]),
                )
            )
        model.load_weights(updates, strict=False)
        loaded_count += len(updates)

    if loaded_count != len(destinations):
        raise _fail("Qwen4 direct hydration did not cover every declared destination")
    return loaded_count


_QWEN4_LAYER_COUNT = 48
_QWEN4_EXPERT_COUNT = 512
_QWEN4_HIDDEN_SIZE = 2560
_QWEN4_INTERMEDIATE_SIZE = 640
_QWEN4_PADDED_INTERMEDIATE_SIZE = 768
_QWEN4_IQK_MEMBERS = frozenset({"iq1_s_r4", "iq2_k", "iq2_ks", "iq3_k"})
_QWEN4_EXPERT_IQK_LAYOUTS = frozenset(
    {IQK_LAYOUT_IQK_RELAYOUT, IQK_LAYOUT_QWEN4_STREAM_MAJOR_V1}
)
_QWEN4_PADDED_DOWN_MEMBERS = frozenset({"iq2_k", "iq2_ks", "iq3_k"})
_QWEN4_ZERO_COUNT_MEAN_POLICY = (
    "same_layer_projection_mean_normalized_route_active_v1"
)
_QWEN4_EARLY_IQ3_ZERO_COUNT_PAIRS = (
    (0, 181),
    (0, 193),
    (0, 236),
    (0, 244),
    (0, 271),
    (0, 413),
    (0, 424),
    (0, 477),
    (1, 116),
)
_QWEN4_QSA_LAYERS = 12
_QWEN4_DEFAULT_CONTEXT_TOKENS = 131_072
_QWEN4_DIRECT_FORMATS = frozenset({"iqk", "kquant", "raw_dtype_passthrough"})
_QWEN4_REQUIRED_OPS = frozenset({"kquant_dequant", "iqk_dequant", "raw_dtype_passthrough"})
_QWEN4_STOP_IDS = frozenset({248044, 248046})


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise _fail(f"{field} must be a positive integer")
    return value


def _zero_count_experts(value: object, *, field: str) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise _fail(f"{field} must be a nonempty expert array")
    if any(
        isinstance(expert, bool)
        or not isinstance(expert, int)
        or not 0 <= expert < _QWEN4_EXPERT_COUNT
        for expert in value
    ):
        raise _fail(f"{field} contains an invalid expert index")
    if value != sorted(set(value)):
        raise _fail(f"{field} must be sorted and unique")
    return tuple(value)


def _shape(value: object, *, field: str) -> tuple[int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value)
    ):
        raise _fail(f"{field} must be [out_features, in_features]")
    return int(value[0]), int(value[1])


def _validate_qwen4_manifest_envelope(manifest: Mapping[str, object]) -> None:
    if manifest.get("artifact_kind") != "package_manifest":
        raise _fail("Qwen4 runtime requires a package_manifest artifact")
    if manifest.get("status") != "valid":
        raise _fail("Qwen4 package manifest is not valid")
    required_ops = manifest.get("required_ops")
    if (
        not isinstance(required_ops, list)
        or any(not isinstance(item, str) for item in required_ops)
        or set(required_ops) != _QWEN4_REQUIRED_OPS
        or required_ops != sorted(_QWEN4_REQUIRED_OPS)
    ):
        raise _fail(
            f"released Qwen4 IQ_K package required_ops must be {sorted(_QWEN4_REQUIRED_OPS)}"
        )


def _qwen4_stop_ids(
    manifest: Mapping[str, object],
    package_dir: Path,
    config: Mapping[str, object],
) -> set[int]:
    tokenizer = manifest.get("tokenizer")
    files = tokenizer.get("files") if isinstance(tokenizer, Mapping) else None
    if not isinstance(files, list):
        raise _fail("Qwen4 tokenizer manifest must declare package files")
    generation = None
    for index, entry in enumerate(files):
        if not isinstance(entry, Mapping):
            raise _fail(f"tokenizer.files[{index}] must be an object")
        if entry.get("path") == "generation_config.json":
            if generation is not None:
                raise _fail("Qwen4 tokenizer duplicates generation_config.json")
            generation = entry
    if generation is None:
        raise _fail("Qwen4 package is missing generation_config.json")
    path = _declared_shard(package_dir, generation.get("path"))
    size = generation.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise _fail("generation_config.json has an invalid declared size")
    if path.stat().st_size != size:
        raise _fail("generation_config.json size does not match the tokenizer manifest")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise _fail("could not read package generation_config.json") from exc
    if not isinstance(payload, Mapping):
        raise _fail("generation_config.json must contain an object")

    stop_ids: set[int] = set()

    def collect(value: object, *, field: str) -> None:
        values = value if isinstance(value, list) else [value]
        for token in values:
            if isinstance(token, bool) or not isinstance(token, int) or token < 0:
                raise _fail(f"{field} must contain nonnegative integer token ids")
            stop_ids.add(token)

    collect(config.get("eos_token_id"), field="architecture.config.eos_token_id")
    collect(payload.get("eos_token_id"), field="generation_config.eos_token_id")
    if stop_ids != _QWEN4_STOP_IDS:
        raise _fail(
            f"released Qwen4 stop ids must be {sorted(_QWEN4_STOP_IDS)}, got {sorted(stop_ids)}"
        )
    return stop_ids


def _qwen4_expert_entries(
    manifest: Mapping[str, object],
    expert_layout=None,
) -> dict[tuple[int, str], Mapping[str, object]]:
    tensors = manifest.get("tensors")
    if not isinstance(tensors, list):
        raise _fail("manifest tensors must be an array")
    entries: dict[tuple[int, str], Mapping[str, object]] = {}
    projections = {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}
    for index, raw in enumerate(tensors):
        if not isinstance(raw, Mapping):
            raise _fail(f"manifest tensor {index} must be an object")
        if raw.get("kind") != "expert":
            if raw.get("format") not in _QWEN4_DIRECT_FORMATS:
                raise _fail(
                    f"{raw.get('source_name')}: unsupported released Qwen4 direct "
                    f"format {raw.get('format')!r}"
                )
            continue
        layer = raw.get("layer_index")
        projection_name = raw.get("projection")
        if (
            isinstance(layer, bool)
            or not isinstance(layer, int)
            or not 0 <= layer < _QWEN4_LAYER_COUNT
            or projection_name not in projections
        ):
            raise _fail(f"manifest expert tensor {index} has invalid ownership")
        projection = projections[str(projection_name)]
        key = layer, projection
        if key in entries:
            raise _fail(f"duplicate Qwen4 expert entry for layer {layer} {projection}")
        expected_module = f"layers.{layer}.mlp.experts.{projection}"
        if (
            raw.get("module_path") != expected_module
            or raw.get("module_weight_key") != f"{expected_module}.weight"
        ):
            raise _fail(f"layer {layer} {projection}: manifest module ownership is inconsistent")
        params = raw.get("format_params")
        if not isinstance(params, Mapping):
            raise _fail(f"layer {layer} {projection}: format_params must be an object")
        logical = _shape(
            params.get("logical_shape"),
            field=f"layer {layer} {projection}.logical_shape",
        )
        stored = _shape(
            params.get("stored_shape"),
            field=f"layer {layer} {projection}.stored_shape",
        )
        expected_logical = (
            (_QWEN4_HIDDEN_SIZE, _QWEN4_INTERMEDIATE_SIZE)
            if projection == "down_proj"
            else (_QWEN4_INTERMEDIATE_SIZE, _QWEN4_HIDDEN_SIZE)
        )
        if logical != expected_logical:
            raise _fail(
                f"layer {layer} {projection}: logical shape {logical} != {expected_logical}"
            )
        declared_zero_experts = _zero_count_experts(
            params.get("zero_count_experts"),
            field=f"layer {layer} {projection}.zero_count_experts",
        )
        if (declared_zero_experts is None) != (
            params.get("zero_count_fallback_policy") is None
        ):
            raise _fail(
                f"layer {layer} {projection}: zero-count expert and policy fields "
                "must be declared together"
            )
        if declared_zero_experts is not None and (
            params.get("zero_count_fallback_policy")
            != _QWEN4_ZERO_COUNT_MEAN_POLICY
            or params.get("calibration_policy") != "per_expert_route_active"
        ):
            raise _fail(
                f"layer {layer} {projection}: zero-count fallback policy drifted"
            )
        if layer < 2:
            conservative = (
                raw.get("format") == "kquant"
                and params.get("kquant_codec") == "q8_0"
                and stored == logical
                and params.get("zero_padding") == 0
            )
            member = params.get("iqk_codec")
            specialized_stored = (
                (_QWEN4_HIDDEN_SIZE, _QWEN4_PADDED_INTERMEDIATE_SIZE)
                if projection == "down_proj" and member in _QWEN4_PADDED_DOWN_MEMBERS
                else logical
            )
            specialized = (
                raw.get("format") == "iqk"
                and member in _QWEN4_IQK_MEMBERS
                and params.get("layout") in _QWEN4_EXPERT_IQK_LAYOUTS
                and stored == specialized_stored
                and params.get("zero_padding")
                == (specialized_stored[1] - logical[1])
                and params.get("calibration_policy") == "per_expert_route_active"
            )
            if not conservative and not specialized:
                raise _fail(
                    f"layer {layer} {projection}: calibration holes require either "
                    "native-width q8_0 or a declared full-IQK specialization"
                )
            if conservative and declared_zero_experts is not None:
                raise _fail(
                    f"layer {layer} {projection}: q8_0 fallback declares an IQ_K "
                    "zero-count specialization"
                )
        else:
            member = params.get("iqk_codec")
            expected_stored = (
                (_QWEN4_HIDDEN_SIZE, _QWEN4_PADDED_INTERMEDIATE_SIZE)
                if projection == "down_proj" and member in _QWEN4_PADDED_DOWN_MEMBERS
                else logical
            )
            expected_padding = expected_stored[1] - logical[1]
            if (
                raw.get("format") != "iqk"
                or member not in _QWEN4_IQK_MEMBERS
                or params.get("layout") not in _QWEN4_EXPERT_IQK_LAYOUTS
                or stored != expected_stored
                or params.get("zero_padding") != expected_padding
            ):
                raise _fail(f"layer {layer} {projection}: IQ_K stored geometry is inconsistent")
        entries[key] = raw
    expected = {
        (layer, projection)
        for layer in range(_QWEN4_LAYER_COUNT)
        for projection in projections.values()
    }
    if set(entries) != expected:
        missing = sorted(expected - set(entries))
        extra = sorted(set(entries) - expected)
        raise _fail(
            "Qwen4 package must declare exactly 144 expert cells "
            f"(missing={missing[:6]} extra={extra[:6]})"
        )
    early = [
        entries[(layer, projection)]
        for layer in (0, 1)
        for projection in projections.values()
    ]
    specialized = any(entry.get("format") == "iqk" for entry in early)
    full_early_iq3 = False
    if specialized:
        if not all(entry.get("format") == "iqk" for entry in entries.values()):
            raise _fail("Qwen4 zero-count specialization must cover all 144 expert cells")
        if expert_layout is None:
            all_iq2 = all(
                entry.get("format_params", {}).get("iqk_codec") == "iq2_k"
                for entry in entries.values()
            )
            full_early_iq3 = all(
                entry.get("format_params", {}).get("iqk_codec")
                == ("iq3_k" if layer < 2 else "iq2_k")
                for (layer, _projection), entry in entries.items()
            )
            if not all_iq2 and not full_early_iq3:
                raise _fail(
                    "full Qwen4 zero-count specialization must use IQ2_K for all "
                    "144 cells or IQ3_K for the six layer 0/1 cells and IQ2_K "
                    "for the other 138 cells"
                )
    zero_sets = {}
    for layer in range(_QWEN4_LAYER_COUNT):
        layer_sets = {
            tuple(entry["format_params"].get("zero_count_experts", []))
            for projection in projections.values()
            for entry in (entries[(layer, projection)],)
        }
        if len(layer_sets) != 1:
            raise _fail(
                f"layer {layer} zero-count expert declarations differ by projection"
            )
        zero_sets[layer] = next(iter(layer_sets))
    declared_pairs = [
        (layer, expert)
        for layer, experts in zero_sets.items()
        for expert in experts
    ]
    if specialized and not declared_pairs:
        raise _fail("Qwen4 zero-count specialization declares no fallback experts")
    if full_early_iq3 and tuple(declared_pairs) != _QWEN4_EARLY_IQ3_ZERO_COUNT_PAIRS:
        raise _fail("full early-IQ3_K specialization zero-count pairs drifted")
    if not specialized and declared_pairs:
        raise _fail("Qwen4 conservative expert package declares zero-count specialization")
    if specialized and expert_layout is not None:
        retained = {
            (layer, expert)
            for layer, row in expert_layout.layers.items()
            for expert in row.source_expert_ids
        }
        leaked = [pair for pair in declared_pairs if pair in retained]
        if leaked:
            raise _fail(
                f"Qwen4 compact expert layout retains unobserved calibration experts {leaked}"
            )
    return entries


def _validate_qwen4_expert_index(
    index: Any,
    entries: Mapping[tuple[int, str], Mapping[str, object]],
    expert_layout=None,
) -> None:
    layers = tuple(index.layers_indexed())
    if layers != tuple(range(_QWEN4_LAYER_COUNT)):
        raise _fail("Qwen4 expert bundles must cover exactly layers 0 through 47")
    problems = index.validate()
    if problems:
        raise _fail("invalid Qwen4 expert index: " + "; ".join(problems))
    for layer in layers:
        expected_experts = (
            _QWEN4_EXPERT_COUNT
            if expert_layout is None
            else expert_layout.layers[layer].num_experts
        )
        if index.num_experts_for_layer(layer) != expected_experts:
            raise _fail(
                f"layer {layer} expert bundle count does not match the manifest layout"
            )
        for projection in ("gate_proj", "up_proj", "down_proj"):
            entry = entries[(layer, projection)]
            params = entry["format_params"]
            assert isinstance(params, Mapping)
            geometry = index.geometry(layer=layer, projection=projection)
            expected_codec = str(entry["format"])
            if geometry.codec != expected_codec:
                raise _fail(
                    f"layer {layer} {projection}: bundle codec {geometry.codec!r} "
                    f"does not match manifest {expected_codec!r}"
                )
            if expected_codec == "kquant":
                member = geometry.kquant_codec
                declared_member = params.get("kquant_codec")
                stored_in = (
                    geometry.packed_cols
                    // int(geometry.bytes_per_block or 1)
                    * int(geometry.weights_per_block or 0)
                )
            else:
                member = geometry.iqk_codec
                declared_member = params.get("iqk_codec")
                stored_in = int(geometry.in_features or 0)
                declared_layout = params.get("layout")
                if geometry.layout not in _QWEN4_EXPERT_IQK_LAYOUTS:
                    raise _fail(
                        f"layer {layer} {projection}: unsupported IQ_K expert layout "
                        f"{geometry.layout!r}"
                    )
                if geometry.layout != declared_layout:
                    raise _fail(
                        f"layer {layer} {projection}: bundle layout {geometry.layout!r} "
                        f"does not match manifest {declared_layout!r}"
                    )
            if member != declared_member:
                raise _fail(
                    f"layer {layer} {projection}: bundle member {member!r} "
                    f"does not match manifest {declared_member!r}"
                )
            stored = _shape(
                params.get("stored_shape"),
                field=f"layer {layer} {projection}.stored_shape",
            )
            if geometry.out_features != stored[0] or stored_in != stored[1]:
                raise _fail(
                    f"layer {layer} {projection}: bundle geometry does not match "
                    "manifest stored_shape"
                )


def validate_qwen4_expert_package_contract(
    manifest: Mapping[str, object],
    index: Any,
) -> dict[tuple[int, str], Mapping[str, object]]:
    """Validate all 144 manifest cells against package bundle metadata."""
    from moespresso.runtime.qwen4.expert_layout import (
        Qwen4ExpertLayoutError,
        parse_qwen4_expert_layout,
        validate_expert_index_counts,
    )

    try:
        expert_layout = parse_qwen4_expert_layout(manifest)
        if expert_layout is not None:
            validate_expert_index_counts(expert_layout, index)
    except Qwen4ExpertLayoutError as exc:
        raise _fail(f"invalid Qwen4 compact expert layout: {exc}") from exc
    entries = _qwen4_expert_entries(manifest, expert_layout)
    _validate_qwen4_expert_index(index, entries, expert_layout)
    return entries


def validate_qwen4_direct_parameter_coverage(
    model: nn.Module,
    manifest: Mapping[str, object],
) -> None:
    """Fail unless the manifest owns every non-expert graph parameter once."""
    tensors = manifest.get("tensors")
    if not isinstance(tensors, list):
        raise _fail("manifest tensors must be an array")
    declared = set()
    for index, entry in enumerate(tensors):
        if not isinstance(entry, Mapping):
            raise _fail(f"manifest tensor {index} must be an object")
        if entry.get("kind") == "expert":
            continue
        for _source, destination in _direct_keys(entry):
            if destination in declared:
                raise _fail(f"duplicate Qwen4 direct destination: {destination}")
            declared.add(destination)
    actual = {name for name, _value in tree_flatten(model.parameters())}
    if actual != declared:
        missing = sorted(actual - declared)
        extra = sorted(declared - actual)
        raise _fail(
            "Qwen4 direct manifest does not cover the constructed graph "
            f"(missing={missing[:6]} extra={extra[:6]})"
        )


def qwen4_kvarn_runtime_bytes(max_context_tokens: int) -> int:
    """Allocated K4/V4 bytes for twelve QSA layers with default int64 positions."""
    from moespresso.runtime.qwen4.kvarn_cache import (
        QWEN38_KVARN_EXACT_SINK,
        qsa_kvarn_partition,
    )
    from moespresso.runtime.qwen4.kvarn_layout import QWEN38_KVARN_K4V4_G128
    from moespresso.runtime.qwen4.kvarn_mutable import (
        QWEN38_KVARN_EXACT_TAIL_CAPACITY,
    )

    tokens = _positive_int(max_context_tokens, field="max_context_tokens")
    layout = QWEN38_KVARN_K4V4_G128
    sink_end, body_end, _ = qsa_kvarn_partition(tokens)
    records = max(0, body_end - sink_end) // layout.tile_tokens
    index_groups = tokens // 4
    per_layer = (
        records * layout.tile_record_bytes
        + 2 * 2 * QWEN38_KVARN_EXACT_SINK * layout.head_dim * 2
        + 2 * 2 * QWEN38_KVARN_EXACT_TAIL_CAPACITY * layout.head_dim * 2
        + index_groups * 128 * 2
        + 3 * index_groups * 8
        + 3 * 128 * 2
        + 3 * 3 * 8
    )
    return _QWEN4_QSA_LAYERS * per_layer


def _qwen4_non_routed_payload_bytes(package_dir: Path) -> int:
    """Count resident model-shard payload while excluding Qwen4 expert bundles."""
    from moespresso.inventory.safetensors_header import read_headers_with_offsets

    total = 0
    expert_bundles = 0
    for shard in sorted(package_dir.glob("model-*.safetensors")):
        for tensor in read_headers_with_offsets(shard):
            if tensor.name.endswith(".experts.tq_bundle"):
                expert_bundles += 1
                continue
            total += tensor.end - tensor.begin
    if expert_bundles != _QWEN4_LAYER_COUNT:
        raise _fail(
            f"Qwen4 package has {expert_bundles} routed bundle tensors; "
            f"expected {_QWEN4_LAYER_COUNT}"
        )
    return total


def _qwen4_capacity_budget(
    *,
    index: Any,
    package_dir: Path,
    max_context_tokens: int,
    resolution_out: dict[str, object] | None = None,
):
    from moespresso.runtime.streaming_capacity import (
        CapacityBudget,
        bytes_per_capacity_unit,
        full_resident_expert_bytes,
        min_capacity,
    )
    from moespresso.runtime.ssd_streaming_build import _resolved_available_bytes

    available_bytes, resolution = _resolved_available_bytes(
        strict_live_available=True,
    )
    if resolution_out is not None:
        resolution_out.update(resolution)

    return CapacityBudget(
        available_bytes=available_bytes,
        resident_base_bytes=_qwen4_non_routed_payload_bytes(package_dir),
        runtime_resident_bytes=qwen4_kvarn_runtime_bytes(max_context_tokens),
        kv_activation_allowance_bytes=int(
            float(os.environ.get("MOESPRESSO_SSD_KV_ALLOWANCE_GB", "2")) * (1 << 30)
        ),
        safety_margin_bytes=int(
            float(os.environ.get("MOESPRESSO_SSD_SAFETY_MARGIN_GB", "2")) * (1 << 30)
        ),
        bytes_per_capacity_unit=bytes_per_capacity_unit(index),
        min_capacity=min_capacity(max_router_fanout=10),
        max_capacity=_QWEN4_EXPERT_COUNT,
        full_resident_expert_bytes=full_resident_expert_bytes(index),
    )


@dataclass
class Qwen4RuntimeResources:
    """Package-backed resources whose lifetime is the loaded graph's lifetime."""

    ple_provider: Any
    expert_executors: tuple[Any, ...]
    closed: bool = False

    def _unique_resources(self) -> tuple[Any, ...]:
        seen = set()
        resources = []
        for resource in (*self.expert_executors, self.ple_provider):
            if id(resource) in seen:
                continue
            seen.add(id(resource))
            resources.append(resource)
        return tuple(resources)

    def assert_quiescent(self) -> None:
        """Preflight every expert executor before teardown mutates anything."""
        if self.closed:
            return
        resources = self._unique_resources()
        for resource in resources:
            assert_quiescent = getattr(resource, "assert_quiescent", None)
            if callable(assert_quiescent):
                assert_quiescent()

    def close(self) -> None:
        if self.closed:
            return
        resources = self._unique_resources()
        self.assert_quiescent()
        for resource in resources:
            close = getattr(resource, "close", None)
            if callable(close):
                close()
        self.closed = True


def load_qwen4_iqk_package_model(
    manifest: Mapping[str, object],
    package_dir: str | Path,
    *,
    capacity_per_layer: int | None = None,
    capacity_overrides: Mapping[int, int] | None = None,
    additional_resident_bytes: int = 0,
    max_context_tokens: int = _QWEN4_DEFAULT_CONTEXT_TOKENS,
    eviction_policy: str = "lfu",
    cache_routing: str = "auto",
    cache_routing_factor: float | None = None,
    cache_routing_protected_routes: int | None = None,
    build_index_fn=None,
    expert_builder=None,
    graph_builder=None,
    install_iqk_dense_fn=None,
    install_kquant_fn=None,
    validate_direct_coverage_fn=None,
    hydrate_fn=None,
    parse_ple_fn=None,
    ple_provider_factory=None,
    load_tokenizer_fn=None,
    seed_expert_residency_fn=None,
):
    """Load one released Qwen4 IQ_K package into a single pooled graph.

    Full residency is the ``capacity=512`` instance of the same expert graph
    used for bounded residency. PLE rows remain package-backed, and every QSA
    layer creates request-owned mutable K4/V4 state on first use.
    """
    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be a mapping")
    from moespresso.runtime.qwen4.cache_routing_config import resolve_cache_routing

    routing = resolve_cache_routing(
        cache_routing, factor=cache_routing_factor, protected_routes=cache_routing_protected_routes,
    )
    max_context_tokens = _positive_int(
        max_context_tokens,
        field="max_context_tokens",
    )
    if max_context_tokens < 4:
        raise _fail("max_context_tokens must be at least 4 for Qwen sparse attention")
    _validate_qwen4_manifest_envelope(manifest)
    root = Path(package_dir)

    if build_index_fn is None:
        from moespresso.runtime.expert_index import build_expert_index

        build_index_fn = build_expert_index
    index = build_index_fn(root)
    validate_qwen4_expert_package_contract(manifest, index)

    architecture = manifest.get("architecture")
    if not isinstance(architecture, Mapping):
        raise _fail("architecture must be an object")
    config = architecture.get("config")
    if not isinstance(config, Mapping):
        raise _fail("architecture.config must be an object")
    package_context_limit = _positive_int(
        config.get("max_position_embeddings"),
        field="architecture.config.max_position_embeddings",
    )
    if max_context_tokens > package_context_limit:
        raise _fail(
            f"max_context_tokens {max_context_tokens} exceeds package limit {package_context_limit}"
        )
    from moespresso.runtime.qwen4.ple_contract import (
        Qwen4PLEProviderError,
        derive_qwen4_ple_provider_contract,
    )

    try:
        ple_contract = derive_qwen4_ple_provider_contract(architecture)
    except Qwen4PLEProviderError as exc:
        raise _fail(f"invalid Qwen4 PLE architecture: {exc}") from exc
    if ple_contract is None:
        raise _fail("released Qwen4 package has no PLE contract")
    if parse_ple_fn is None:
        from moespresso.runtime.qwen4.ple_provider import parse_qwen4_ple_provider

        parse_ple_fn = parse_qwen4_ple_provider
    if ple_provider_factory is None:
        from moespresso.runtime.qwen4.ple_provider import Qwen4PLEDirectRowProvider

        ple_provider_factory = Qwen4PLEDirectRowProvider
    layout = parse_ple_fn(manifest, root, expected=ple_contract)
    ple_provider = ple_provider_factory(layout)
    executors: list[Any] = []
    resources = Qwen4RuntimeResources(ple_provider, ())
    try:
        if capacity_per_layer is None:
            from moespresso.runtime.streaming_capacity import choose_capacity

            budget_resolution: dict[str, object] = {}
            budget = _qwen4_capacity_budget(
                index=index,
                package_dir=root,
                max_context_tokens=max_context_tokens,
                resolution_out=budget_resolution,
            )
            if type(additional_resident_bytes) is not int or additional_resident_bytes < 0:
                raise _fail("additional_resident_bytes must be a nonnegative integer")
            budget = replace(budget, runtime_resident_bytes=budget.runtime_resident_bytes + additional_resident_bytes)
            capacity_per_layer = choose_capacity(budget)
            capacity_budget = {
                "available_bytes": budget.available_bytes,
                "resident_base_bytes": budget.resident_base_bytes,
                "runtime_resident_bytes": budget.runtime_resident_bytes,
                "additional_resident_bytes": additional_resident_bytes,
                "kv_activation_allowance_bytes": budget.kv_activation_allowance_bytes,
                "safety_margin_bytes": budget.safety_margin_bytes,
                "bytes_per_capacity_unit": budget.bytes_per_capacity_unit,
                "usable_bytes": budget.usable_bytes,
                "min_capacity": budget.min_capacity,
                "max_capacity": budget.max_capacity,
                "full_resident_expert_bytes": budget.full_resident_expert_bytes,
                "planner_resolution": budget_resolution,
            }
        else:
            capacity_per_layer = _positive_int(
                capacity_per_layer,
                field="capacity_per_layer",
            )
            capacity_budget = None
        if capacity_per_layer > _QWEN4_EXPERT_COUNT:
            raise _fail("capacity_per_layer cannot exceed 512 experts")
        overrides = dict(capacity_overrides or {})
        if any(
            isinstance(layer, bool)
            or not isinstance(layer, int)
            or not 0 <= layer < _QWEN4_LAYER_COUNT
            or isinstance(capacity, bool)
            or not isinstance(capacity, int)
            or capacity <= 0
            or capacity > _QWEN4_EXPERT_COUNT
            for layer, capacity in overrides.items()
        ):
            raise _fail("capacity_overrides must map released layers to positive integers")
        if expert_builder is None:
            from moespresso.runtime.qwen4.expert_provider import (
                build_qwen4_pooled_expert_executor,
            )

            expert_builder = build_qwen4_pooled_expert_executor

        def expert_factory(layer: int, hidden: int, width: int, experts: int):
            executor = expert_builder(
                package_dir=root,
                index=index,
                layer=layer,
                hidden_size=hidden,
                intermediate_size=width,
                num_experts=experts,
                capacity=int(overrides.get(layer, capacity_per_layer)),
                eviction_policy=eviction_policy,
            )
            executors.append(executor)
            resources.expert_executors = tuple(executors)
            return executor

        if graph_builder is None:
            from moespresso.runtime.qwen4.build import build_qwen4_graph_from_manifest

            graph_builder = build_qwen4_graph_from_manifest
        from moespresso.runtime.qwen4.qsa_kvarn import (
            Qwen4MutableKVarNQSAStateBackend,
        )

        model = graph_builder(
            manifest,
            expert_factory=expert_factory,
            ple_provider=ple_provider,
            qsa_state_backend_factory=lambda module: Qwen4MutableKVarNQSAStateBackend(
                module,
                max_context_tokens=max_context_tokens,
            ),
            qsa_max_query_tokens=64,
        )
        if len(executors) != _QWEN4_LAYER_COUNT:
            raise _fail(
                f"Qwen4 graph constructed {len(executors)} expert executors; "
                f"expected {_QWEN4_LAYER_COUNT}"
            )
        from moespresso.runtime.pooled_moe import install_pooled_decode_session

        install_pooled_decode_session(model, executors)
        object.__setattr__(model, "_moespresso_owns_pooled_request_scope", True)
        object.__setattr__(
            model,
            "_moespresso_pooled_decode_bounded",
            any(
                pool.capacity < pool.num_experts
                for executor in executors
                for pool in executor._projection_pools_lockstep()
            ),
        )
        model.layers[-1].mlp.pipeline_is_last = True
        if install_iqk_dense_fn is None:
            from moespresso.runtime.qwen4.iqk_dense import (
                install_qwen4_iqk_dense_modules,
            )

            install_iqk_dense_fn = install_qwen4_iqk_dense_modules
        installed_iqk = int(install_iqk_dense_fn(model, manifest))
        direct_iqk = sum(
            1
            for tensor in manifest["tensors"]
            if isinstance(tensor, Mapping)
            and tensor.get("kind") != "expert"
            and tensor.get("format") == "iqk"
        )
        if installed_iqk != direct_iqk:
            raise _fail(
                f"installed {installed_iqk} direct IQ_K modules; manifest declares {direct_iqk}"
            )
        if install_kquant_fn is None:
            from moespresso.runtime.kquant_install import install_manifest_kquant_modules

            install_kquant_fn = install_manifest_kquant_modules
        installed = int(install_kquant_fn(model, dict(manifest)))
        direct_kquant = sum(
            1
            for tensor in manifest["tensors"]
            if isinstance(tensor, Mapping)
            and tensor.get("kind") != "expert"
            and tensor.get("format") == "kquant"
        )
        if installed != direct_kquant:
            raise _fail(
                f"installed {installed} direct K-quant modules; manifest declares {direct_kquant}"
            )
        if validate_direct_coverage_fn is None:
            validate_direct_coverage_fn = validate_qwen4_direct_parameter_coverage
        validate_direct_coverage_fn(model, manifest)
        if hydrate_fn is None:
            hydrate_fn = hydrate_qwen4_package_weights
        hydrate_fn(model, manifest, root)
        mx.eval(model.parameters())
        model.eval()
        if seed_expert_residency_fn is None:
            from moespresso.runtime.ssd_streaming_build import (
                seed_expert_residency as seed_expert_residency_fn,
            )
        hotlist_info = seed_expert_residency_fn(model, root)
        object.__setattr__(model, "_moespresso_ssd_hotlist", hotlist_info)
        if routing.policy == "auto":
            from moespresso.runtime.qwen4.cache_routing_config import (
                resolve_auto_cache_routing,
            )

            routing = resolve_auto_cache_routing(
                bounded=bool(model._moespresso_pooled_decode_bounded),
            )
            routing_resolution = (
                "auto-bounded" if routing.enabled else "auto-full-resident"
            )
        else:
            routing_resolution = "explicit"
        if routing.enabled:
            from moespresso.runtime.qwen4.cache_routing import configure_cache_routing

            configure_cache_routing(
                model, routing.policy, factor=routing.cache_factor,
                protected_routes=routing.protected_routes,
            )
        object.__setattr__(
            model,
            "_moespresso_cache_routing_resolution",
            routing_resolution,
        )
        if load_tokenizer_fn is None:
            from mlx_lm.utils import load_tokenizer

            load_tokenizer_fn = load_tokenizer
        stop_ids = _qwen4_stop_ids(manifest, root, config)
        tokenizer = load_tokenizer_fn(root, eos_token_ids=stop_ids)
        from moespresso.runtime.qwen4.generation import (
            QWEN4_GENERATION_ADAPTER,
        )

        object.__setattr__(
            model,
            "_moespresso_generation_adapter",
            QWEN4_GENERATION_ADAPTER,
        )
        object.__setattr__(model, "_moespresso_qwen4_stop_ids", stop_ids)
        object.__setattr__(model, "_moespresso_qwen4_runtime_resources", resources)
        object.__setattr__(model, "_moespresso_ssd_streaming_capacity", int(capacity_per_layer))
        object.__setattr__(
            model,
            "_moespresso_ssd_streaming_resolved_capacities",
            {
                layer: int(getattr(executor, "resolved_capacity", capacity_per_layer))
                for layer, executor in enumerate(executors)
            },
        )
        object.__setattr__(model, "_moespresso_ssd_streaming_capacity_overrides", overrides)
        object.__setattr__(model, "_moespresso_ssd_streaming_eviction_policy", eviction_policy)
        object.__setattr__(model, "_moespresso_qwen4_kvarn_context_tokens", max_context_tokens)
        object.__setattr__(
            model,
            "_moespresso_qwen4_kvarn_runtime_bytes",
            qwen4_kvarn_runtime_bytes(max_context_tokens),
        )
        if capacity_budget is not None:
            object.__setattr__(model, "_moespresso_ssd_streaming_capacity_budget", capacity_budget)
        return model, tokenizer
    except BaseException:
        resources.close()
        raise
