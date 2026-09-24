"""Build the SSD-streaming MoE runtime.

The builder instantiates the MLX model skeleton, replaces routed SwitchGLU
experts with codec-aware persistent pools, then loads only non-routed tensors
as regular model parameters. Full pool capacity is the all-resident subcase;
smaller pools load bundle rows on demand. Resident loaders that materialize a
separate routed stack are not used here.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path

import mlx.core as mx

from moespresso.runtime.build import (
    _qwen_eos_token_ids,
    _silence_known_transformers_warnings,
)
from moespresso.runtime.expert_index import ExpertIndex, build_expert_index
from moespresso.runtime.expert_slot_pool import BundleRowCache
from moespresso.runtime.pooled_switchglu import (
    PooledDeepseekV4MoEBlock,
    PooledCombinedGateUpKQuantLinear,
    PooledIqkSwitchLinear,
    PooledKQuantSwitchLinear,
    PooledMxfp4SwitchLinear,
    PooledSparseMoeBlock,
    PooledSwitchGLU,
)
from moespresso.package.bundle import IQK_CODEC, KQUANT_CODEC, MXFP4_CODEC
from moespresso.runtime.streaming_capacity import (
    available_memory_bytes,
    choose_capacity,
    is_routed_expert_payload_key,
    package_capacity_budget,
)

_SWITCH_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


class SSDStreamingBuildError(RuntimeError):
    pass


def _load_qwen_streaming_tokenizer(
    package_dir: Path,
    config: dict,
    *,
    load_tokenizer_fn=None,
):
    if load_tokenizer_fn is None:
        from mlx_lm.utils import load_tokenizer

        load_tokenizer_fn = load_tokenizer
    return load_tokenizer_fn(
        package_dir,
        eos_token_ids=_qwen_eos_token_ids(config),
    )


def _layers(model) -> list:
    """Return the model's decoder layers, handling wrapper and text-only shapes."""
    candidates = (
        getattr(getattr(getattr(model, "language_model", None), "model", None), "layers", None),
        getattr(getattr(model, "model", None), "layers", None),
        getattr(model, "layers", None),
    )
    for layers in candidates:
        if layers is not None:
            return layers
    raise SSDStreamingBuildError("could not find decoder layers on model")


def _pooled_switch_for_layer(layer) -> PooledSwitchGLU | None:
    """Return the routed pooled executor for supported decoder layouts."""
    mlp = getattr(layer, "mlp", None)
    for name in ("switch_mlp", "experts"):
        switch = getattr(mlp, name, None)
        if isinstance(switch, PooledSwitchGLU):
            return switch
    return None


def _is_routed_expert_key(key: str) -> bool:
    return is_routed_expert_payload_key(key)


def _projection_dims(sw, projection: str) -> tuple[int, int]:
    proj = getattr(sw, projection, None)
    if proj is None:
        raise SSDStreamingBuildError(f"switch_mlp is missing {projection}")

    in_features = getattr(proj, "input_dims", None)
    out_features = getattr(proj, "output_dims", None)
    if in_features is None:
        in_features = getattr(proj, "in_features", None)
    if out_features is None:
        out_features = getattr(proj, "out_features", None)
    if in_features is None or out_features is None:
        raise SSDStreamingBuildError(f"could not infer dimensions for switch_mlp.{projection}")
    return int(in_features), int(out_features)


def _pooled_projection(
    *,
    package_dir: Path,
    index: ExpertIndex,
    layer: int,
    projection: str,
    capacity_per_layer: int,
    in_features: int,
    out_features: int,
    eviction_policy: str,
    row_cache=None,
    spare_slots: int = 0,
) -> (
    PooledMxfp4SwitchLinear
    | PooledKQuantSwitchLinear
    | PooledIqkSwitchLinear
):
    geometry = index.geometry(layer=layer, projection=projection)
    bits = geometry.bits
    if geometry.codec == KQUANT_CODEC:
        bytes_per_block = int(geometry.bytes_per_block or 0)
        weights_per_block = int(geometry.weights_per_block or 0)
        if bytes_per_block <= 0 or weights_per_block <= 0:
            raise SSDStreamingBuildError(f"layer {layer} {projection}: missing K-quant geometry")
        if geometry.packed_cols % bytes_per_block:
            raise SSDStreamingBuildError(
                f"layer {layer} {projection}: K-quant bytes_per_row "
                f"{geometry.packed_cols} is not divisible by {bytes_per_block}"
            )
        packed_in_features = geometry.packed_cols // bytes_per_block * weights_per_block
    elif geometry.codec == IQK_CODEC:
        packed_in_features = int(geometry.in_features or 0)
    else:
        packed_in_features = geometry.packed_cols * (32 // bits)
    if packed_in_features != in_features:
        raise SSDStreamingBuildError(
            f"layer {layer} {projection}: skeleton input dim {in_features} "
            f"!= packed geometry dim {packed_in_features}"
        )
    if geometry.out_features != out_features:
        raise SSDStreamingBuildError(
            f"layer {layer} {projection}: skeleton output dim {out_features} "
            f"!= packed geometry dim {geometry.out_features}"
        )

    if geometry.codec == MXFP4_CODEC:
        return PooledMxfp4SwitchLinear(
            package_dir=package_dir,
            index=index,
            layer=layer,
            projection=projection,
            capacity=capacity_per_layer,
            eviction_policy=eviction_policy,
            row_cache=row_cache,
            spare_slots=spare_slots,
        )
    if geometry.codec == KQUANT_CODEC:
        return PooledKQuantSwitchLinear(
            package_dir=package_dir,
            index=index,
            layer=layer,
            projection=projection,
            capacity=capacity_per_layer,
            eviction_policy=eviction_policy,
            row_cache=row_cache,
            spare_slots=spare_slots,
        )
    if geometry.codec == IQK_CODEC:
        return PooledIqkSwitchLinear(
            package_dir=package_dir,
            index=index,
            layer=layer,
            projection=projection,
            capacity=capacity_per_layer,
            eviction_policy=eviction_policy,
            row_cache=row_cache,
            spare_slots=spare_slots,
        )
    raise SSDStreamingBuildError(
        f"layer {layer} {projection}: unsupported expert codec {geometry.codec!r}"
    )


def _can_combine_gate_up_kquant(index: ExpertIndex, *, layer: int) -> bool:
    try:
        gate = index.geometry(layer=layer, projection="gate_proj")
        up = index.geometry(layer=layer, projection="up_proj")
    except KeyError:
        return False
    if gate.codec != KQUANT_CODEC or up.codec != KQUANT_CODEC:
        return False
    return (
        gate.bits == up.bits
        and gate.packed_cols == up.packed_cols
        and gate.packed_dtype == up.packed_dtype
        and gate.kquant_codec == up.kquant_codec
        and gate.group_size == up.group_size
        and gate.bytes_per_block == up.bytes_per_block
        and gate.weights_per_block == up.weights_per_block
    )


def _projection_modules_for_switch(switch) -> tuple[tuple[str, object], ...]:
    return tuple((projection, getattr(switch, projection)) for projection in _SWITCH_PROJECTIONS)


def _unique_projection_pools_for_switch(switch) -> tuple[object, ...]:
    pools = []
    seen = set()
    for _projection, module in _projection_modules_for_switch(switch):
        pool = getattr(module, "pool")
        ident = id(pool)
        if ident in seen:
            continue
        seen.add(ident)
        pools.append(pool)
    return tuple(pools)


def build_pooled_switchglu(
    *,
    package_dir: str | Path,
    index: ExpertIndex,
    layer: int,
    capacity: int,
    projection_dims: Mapping[str, tuple[int, int]],
    activation,
    eviction_policy: str = "lfu",
    spare_slots: int = 0,
) -> PooledSwitchGLU:
    """Build one codec-aware pooled expert executor from package geometry."""
    if capacity < 1:
        raise SSDStreamingBuildError(f"layer {layer} capacity must be >= 1")
    expected = set(_SWITCH_PROJECTIONS)
    if set(projection_dims) != expected:
        raise SSDStreamingBuildError(
            f"layer {layer} projection dimensions must cover {sorted(expected)}"
        )
    normalized_dims: dict[str, tuple[int, int]] = {}
    for projection in _SWITCH_PROJECTIONS:
        dims = tuple(int(value) for value in projection_dims[projection])
        if len(dims) != 2 or any(value <= 0 for value in dims):
            raise SSDStreamingBuildError(
                f"layer {layer} {projection} dimensions must be two positive integers"
            )
        normalized_dims[projection] = dims
    layer_num_experts = index.num_experts_for_layer(layer)
    resolved_capacity = min(int(capacity), layer_num_experts)
    resolved_spare_slots = min(
        int(spare_slots),
        max(0, layer_num_experts - resolved_capacity),
    )
    if resolved_spare_slots < 0:
        raise ValueError("spare_slots must be >= 0")

    combine_gate_up = _can_combine_gate_up_kquant(index, layer=layer)
    row_cache = BundleRowCache(
        package_dir=package_dir,
        index=index,
        layer=layer,
        consumers=2 if combine_gate_up else 3,
    )
    projections = {}
    if combine_gate_up:
        gate_dims = normalized_dims["gate_proj"]
        up_dims = normalized_dims["up_proj"]
        if gate_dims != up_dims:
            raise SSDStreamingBuildError(
                f"layer {layer}: cannot combine gate/up K-quant with "
                f"different skeleton dims gate={gate_dims} up={up_dims}"
            )
        combined = PooledCombinedGateUpKQuantLinear(
            package_dir=package_dir,
            index=index,
            layer=layer,
            capacity=resolved_capacity,
            eviction_policy=eviction_policy,
            row_cache=row_cache,
            spare_slots=resolved_spare_slots,
        )
        projections["gate_proj"] = combined
        projections["up_proj"] = combined.up_alias
        in_features, out_features = normalized_dims["down_proj"]
        projections["down_proj"] = _pooled_projection(
            package_dir=Path(package_dir),
            index=index,
            layer=layer,
            projection="down_proj",
            capacity_per_layer=resolved_capacity,
            in_features=int(in_features),
            out_features=int(out_features),
            eviction_policy=eviction_policy,
            row_cache=row_cache,
            spare_slots=resolved_spare_slots,
        )
    else:
        for projection in _SWITCH_PROJECTIONS:
            in_features, out_features = normalized_dims[projection]
            projections[projection] = _pooled_projection(
                package_dir=Path(package_dir),
                index=index,
                layer=layer,
                projection=projection,
                capacity_per_layer=resolved_capacity,
                in_features=int(in_features),
                out_features=int(out_features),
                eviction_policy=eviction_policy,
                row_cache=row_cache,
                spare_slots=resolved_spare_slots,
            )
    pooled = PooledSwitchGLU(
        gate_proj=projections["gate_proj"],
        up_proj=projections["up_proj"],
        down_proj=projections["down_proj"],
        activation=activation,
    )
    pooled.resolved_capacity = resolved_capacity
    pooled.resolved_spare_slots = resolved_spare_slots
    return pooled


def install_pooled_switchglus(
    model,
    *,
    package_dir: str | Path,
    index: ExpertIndex,
    capacity_per_layer: int,
    capacity_overrides: Mapping[int, int] | None = None,
    eviction_policy: str = "lfu",
    spare_slots: int = 0,
    wrap_deepseek_v4_moe: bool = False,
) -> int:
    """Replace indexed routed SwitchGLU layers with SSD-backed pooled modules."""
    package_dir = Path(package_dir)
    capacity_overrides = dict(capacity_overrides or {})
    problems = index.validate()
    if problems:
        raise SSDStreamingBuildError("invalid expert index: " + "; ".join(problems))

    installed = 0
    last_block = None
    switches = []
    from moespresso.runtime.pooled_moe import install_pooled_decode_session
    resolved_capacities: dict[int, int] = {}
    for layer_idx, layer in enumerate(_layers(model)):
        mlp = getattr(layer, "mlp", None)
        sw = getattr(mlp, "switch_mlp", None)
        if sw is None:
            if layer_idx in index.layers_indexed():
                raise SSDStreamingBuildError(
                    f"expert index has layer {layer_idx}, but model has no switch_mlp"
                )
            continue
        if not index.has_projection(layer=layer_idx, projection="gate_proj"):
            continue

        layer_capacity = int(capacity_overrides.get(layer_idx, capacity_per_layer))
        if layer_capacity < 1:
            raise SSDStreamingBuildError(f"layer {layer_idx} capacity must be >= 1")
        if layer_capacity + spare_slots > index.max_num_experts:
            raise ValueError("capacity+spare_slots cannot exceed num_experts")
        pooled_switch = build_pooled_switchglu(
            package_dir=package_dir,
            index=index,
            layer=layer_idx,
            capacity=layer_capacity,
            projection_dims={
                projection: _projection_dims(sw, projection) for projection in _SWITCH_PROJECTIONS
            },
            activation=sw.activation,
            eviction_policy=eviction_policy,
            spare_slots=spare_slots,
        )
        layer_capacity = pooled_switch.resolved_capacity
        if pooled_switch._all_iqk:
            pooled_switch.iqk_ordinal = installed
        setattr(mlp, "switch_mlp", pooled_switch)
        switches.append(pooled_switch)
        if all(
            hasattr(mlp, name)
            for name in (
                "gate",
                "shared_expert",
                "shared_expert_gate",
            )
        ):
            block = PooledSparseMoeBlock(mlp)
            setattr(layer, "mlp", block)
            last_block = block
        elif wrap_deepseek_v4_moe and all(
            hasattr(mlp, name)
            for name in (
                "gate",
                "shared_experts",
            )
        ):
            block = PooledDeepseekV4MoEBlock(mlp)
            setattr(layer, "mlp", block)
            last_block = block
        resolved_capacities[layer_idx] = layer_capacity
        installed += 1

    if last_block is not None:
        # the pipelined decode joins its worker queue at the deepest MoE layer
        last_block.pipeline_is_last = True

    install_pooled_decode_session(model, switches)

    expected = set(index.layers_indexed())
    if installed != len(expected):
        raise SSDStreamingBuildError(
            f"installed {installed} pooled SwitchGLU layer(s), "
            f"but expert index declares {len(expected)} layer(s)"
        )
    object.__setattr__(
        model,
        "_moespresso_ssd_streaming_resolved_capacities",
        dict(sorted(resolved_capacities.items())),
    )
    return installed


def _read_manifest(package_dir: Path) -> dict | None:
    """Best-effort manifest read for the streaming builder's dense-codec check."""
    manifest_path = package_dir / "package_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        return json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        return None


def _build_bare_skeleton(cfg: Mapping):
    """Construct the mlx-lm model class from config without loading weights.

    The K-quant dense install needs leaves whose shapes are the model's logical
    dims (`nn.Embedding(vocab, hidden)`). Packed byte width in the shard has no
    bearing on the module geometry.
    mlx-lm's `load_model` hydrates leaves at construction, so this builds the
    skeleton the way the resident K-quant path does: instantiate the class from
    config and leave `_load_non_routed_resident` to load the tensors after the
    module swap. The routed experts are replaced by the pooled installer, so
    their placeholder leaves never carry weights either way.
    """
    from mlx_lm.utils import _get_classes

    config = dict(cfg)
    model_class, model_args_class = _get_classes(config=config)
    model = model_class(model_args_class.from_dict(config))
    model.eval()
    return model


def _maybe_install_kquant_dense(model, manifest: dict | None) -> int:
    """Swap resident dense leaves to K-quant modules when the manifest asks.

    Returns the number of installed modules. On a package whose non-expert
    tensors are affine this is a no-op (empty codec map) and returns 0, so
    affine-dense serving is byte-identical to before. On a K-quant dense
    package it installs the same manifest-declared modules the resident path
    installs (`install_manifest_kquant_modules`), including the K-quant
    embedding, before the resident weight load hydrates their wire bytes.

    Fails closed: an unknown K-quant codec or a missing installer key raises
    (through `install_manifest_kquant_modules`) rather than letting uint8 wire
    bytes fall through to a stock affine module.
    """
    if manifest is None:
        return 0
    from moespresso.runtime.kquant_install import (
        install_manifest_kquant_modules,
        kquant_weight_codec_map_from_manifest,
    )

    codec_map = kquant_weight_codec_map_from_manifest(manifest)
    if not codec_map:
        return 0
    installed = install_manifest_kquant_modules(model, manifest)
    print(
        f"[ssd-streaming] installed {installed} K-quant dense module(s) before the resident load.",
        flush=True,
    )
    return installed


def _load_non_routed_resident(model, package_dir: str | Path) -> None:
    """Load every tensor except routed-expert bundles, then materialize it."""
    from jang_tools.ssm_layout import sanitize_grouped_conv1d_layout

    package_dir = Path(package_dir)
    for shard in sorted(package_dir.glob("model-*.safetensors")):
        weights = mx.load(str(shard))
        regular = {key: value for key, value in weights.items() if not _is_routed_expert_key(key)}
        del weights
        if hasattr(model, "sanitize"):
            regular = model.sanitize(regular)
        regular = sanitize_grouped_conv1d_layout(
            regular,
            lambda value: value.moveaxis(2, 1),
        )
        model.load_weights(list(regular.items()), strict=False)
        del regular
    mx.eval(model.parameters())


def _budget_payload(
    budget,
    *,
    resolution: Mapping[str, object] | None = None,
) -> dict:
    payload = {
        "available_bytes": budget.available_bytes,
        "resident_base_bytes": budget.resident_base_bytes,
        "runtime_resident_bytes": budget.runtime_resident_bytes,
        "kv_activation_allowance_bytes": budget.kv_activation_allowance_bytes,
        "safety_margin_bytes": budget.safety_margin_bytes,
        "bytes_per_capacity_unit": budget.bytes_per_capacity_unit,
        "usable_bytes": budget.usable_bytes,
        "min_capacity": budget.min_capacity,
        "max_capacity": budget.max_capacity,
        "full_resident_expert_bytes": budget.full_resident_expert_bytes,
    }
    if resolution is not None:
        payload["planner_resolution"] = dict(resolution)
    return payload


def _resolved_available_bytes(
    *,
    already_resident_bytes: int = 0,
    strict_live_available: bool = False,
    wired_budget_fn=None,
) -> tuple[int, dict[str, object]]:
    """Resolve the startup planner input and report the limiting source."""
    import math
    import os

    import psutil

    from moespresso.runtime.streaming_capacity import usable_wired_budget_bytes

    gib = 1 << 30
    reserve_gb = float(os.environ.get("MOESPRESSO_SSD_OS_RESERVE_GB", "5"))
    if not math.isfinite(reserve_gb) or reserve_gb < 0:
        raise ValueError("MOESPRESSO_SSD_OS_RESERVE_GB must be finite and >= 0")
    already_resident_bytes = int(already_resident_bytes)
    if already_resident_bytes < 0:
        raise ValueError("already_resident_bytes must be >= 0")

    vm = psutil.virtual_memory()
    total_bytes = max(0, int(vm.total))
    physical_ceiling = max(0, int(total_bytes - reserve_gb * gib))
    live_available_bytes = max(0, min(total_bytes, int(vm.available)))
    live_budget = max(
        0,
        min(total_bytes, live_available_bytes + already_resident_bytes),
    )
    cap_gb = os.environ.get("MOESPRESSO_SSD_MAX_MEMORY_GB")
    explicit_ceiling = None
    wired_budget = None
    wired_source = "not-consulted"
    if cap_gb:
        parsed_cap_gb = float(cap_gb)
        if not math.isfinite(parsed_cap_gb) or parsed_cap_gb <= 0:
            raise ValueError("MOESPRESSO_SSD_MAX_MEMORY_GB must be finite and > 0")
        explicit_ceiling = int(parsed_cap_gb * gib)
        automatic_ceiling = None
        candidates = {
            "physical-reserve": physical_ceiling,
            "live-available": live_budget,
            "explicit-max-memory": explicit_ceiling,
        }
    else:
        if wired_budget_fn is None:
            wired_budget_fn = usable_wired_budget_bytes
        wired_budget, wired_source = wired_budget_fn()
        if (
            not isinstance(wired_budget, int)
            or isinstance(wired_budget, bool)
            or wired_budget <= 0
        ):
            wired_budget = None
        automatic_ceiling = (
            max(0, wired_budget - gib) if wired_budget is not None else physical_ceiling
        )
        candidates = {
            "physical-reserve": physical_ceiling,
            "automatic-wired-headroom": automatic_ceiling,
        }
        stable_ceiling = min(physical_ceiling, automatic_ceiling)
        if strict_live_available or live_budget < stable_ceiling * 0.75:
            candidates["live-available"] = live_budget

    limiting_source, resolved = min(candidates.items(), key=lambda item: item[1])
    details: dict[str, object] = {
        "resolved_bytes": int(resolved),
        "limiting_source": limiting_source,
        "strict_live_available": strict_live_available,
        "physical_total_bytes": total_bytes,
        "physical_reserve_ceiling_bytes": physical_ceiling,
        "live_available_bytes": live_available_bytes,
        "already_resident_bytes": already_resident_bytes,
        "live_budget_bytes": live_budget,
        "wired_budget_bytes": wired_budget,
        "wired_budget_source": wired_source,
        "automatic_ceiling_bytes": automatic_ceiling,
        "explicit_ceiling_bytes": explicit_ceiling,
        "automatic_wired_headroom_bytes": gib,
    }
    return int(resolved), details


def _deterministic_available_bytes(*, already_resident_bytes: int = 0) -> int:
    """Return the resolved startup capacity-planner input.

    Automatic planning holds 1 GiB below the reported wired-memory budget
    and also respects the physical-memory reserve. This leaves modest room
    outside the planned package allocation without treating the reported
    recommendation as a root-cause diagnosis. Family loaders can make live
    availability a hard ceiling; other automatic paths retain their pressure
    threshold.

    Some family loaders hydrate the non-routed core before installing expert
    pools. ``already_resident_bytes`` adds that package-owned allocation back
    to the live reading because the capacity budget subtracts it separately.
    An explicit memory ceiling replaces the automatic wired-budget heuristic,
    but remains capped by physical memory and live availability.
    """
    resolved, _details = _resolved_available_bytes(
        already_resident_bytes=already_resident_bytes,
    )
    return resolved


def build_ssd_streaming_model(
    package_dir: str | Path,
    *,
    capacity_per_layer: int | None = None,
    capacity_overrides: Mapping[int, int] | None = None,
    eviction_policy: str = "lfu",
):
    """Build `(model, tokenizer, installed)` with routed experts streamed."""
    from mlx_lm.utils import load_config, load_model, load_tokenizer
    from moespresso.runtime.streaming_run_lock import acquire_ssd_streaming_process_lock

    package_dir = Path(package_dir)
    run_lock = acquire_ssd_streaming_process_lock()
    try:
        _silence_known_transformers_warnings()
        cfg = load_config(package_dir)
        manifest = _read_manifest(package_dir)
        from moespresso.runtime.kquant_install import (
            kquant_weight_codec_map_from_manifest,
        )

        has_kquant_dense = bool(
            manifest is not None
            and kquant_weight_codec_map_from_manifest(manifest)
        )
        if has_kquant_dense:
            # A K-quant dense package needs stock nn.Linear/nn.Embedding leaves
            # so the K-quant module swap can replace them; the affine
            # quantization block would build QuantizedLinear leaves that reject
            # the uint8 wire bytes. This mirrors the resident K-quant path,
            # which also disables the affine block before installing.
            cfg["quantization"] = None
            cfg["quantization_config"] = None
        elif "quantization" not in cfg:
            cfg["quantization"] = {"group_size": 64, "bits": 4}
        text_config = cfg.get("text_config", cfg)

        if has_kquant_dense:
            # mlx_lm.load_model eagerly hydrates the constructed leaves with the
            # shard tensors, so a plain nn.Embedding/nn.Linear would receive the
            # raw K-quant uint8 wire bytes and report a byte-width shape (the
            # K-quant module install derives its geometry from module.weight and
            # would then reject that width). Build the skeleton only, with the
            # config's logical dims, and let install_manifest_kquant_modules swap
            # the leaves before `_load_non_routed_resident` hydrates them. This
            # matches how the resident K-quant path constructs its skeleton.
            model = _build_bare_skeleton(cfg)
        else:
            model, _cfg = load_model(
                package_dir,
                lazy=True,
                strict=False,
                model_config=cfg,
            )
        index = build_expert_index(package_dir)
        budget_payload = None
        if capacity_per_layer is None:
            from moespresso.runtime.qwen.router_gemv import (
                router_bf16_f32_resident_bytes,
            )

            available_bytes, resolution = _resolved_available_bytes()
            budget = package_capacity_budget(
                index=index,
                package_dir=package_dir,
                max_router_fanout=int(text_config.get("num_experts_per_tok", 1)),
                available_bytes=available_bytes,
                runtime_resident_bytes=router_bf16_f32_resident_bytes(cfg),
            )
            capacity_per_layer = choose_capacity(budget)
            budget_payload = _budget_payload(budget, resolution=resolution)
        installed = install_pooled_switchglus(
            model,
            package_dir=package_dir,
            index=index,
            capacity_per_layer=capacity_per_layer,
            capacity_overrides=capacity_overrides,
            eviction_policy=eviction_policy,
        )
        kquant_dense_installed = _maybe_install_kquant_dense(model, manifest)
        object.__setattr__(
            model,
            "_moespresso_ssd_kquant_dense_installed",
            int(kquant_dense_installed),
        )
        _load_non_routed_resident(model, package_dir)

        # Router shadows depend on the hydrated F32 package weights. The
        # installer validates all forty matrices before changing any layer and
        # retains the original linears for prefill and fallback.
        from moespresso.runtime.qwen.router_gemv import (
            install_router_bf16_f32_gemv,
        )

        install_router_bf16_f32_gemv(model)
        hotlist_info = seed_expert_residency(model, package_dir)
        object.__setattr__(model, "_moespresso_ssd_hotlist", hotlist_info)
        # Apply the resident build's long-prompt prefill chunk policy. The
        # default uses a 4096-token chunk for longer prompts and retains the
        # mlx_lm chunk for shorter prompts; MOESPRESSO_QWEN_PREFILL_CHUNK
        # overrides the value. Larger chunks route more token-expert pairs per
        # forward pass within the same residency budget.
        from moespresso.runtime.qwen.prefill_chunk import install_prefill_chunk

        install_prefill_chunk(model)

        # Flash D=256 prefill attention for the full-attention layers, the same
        # route the resident build installs. The pooled MoE swap leaves
        # `self_attn` on the full-attention layers untouched, and the attention
        # weights are resident by this point, so the streamed build wraps the
        # identical modules the resident build wraps. Without the required
        # kernel, the attention module keeps its composed path.
        from moespresso.runtime.qwen.full_attention import (
            install_flash_prefill_attention,
        )

        flash_wrapped = install_flash_prefill_attention(model)
        object.__setattr__(model, "_moespresso_ssd_flash_prefill_layers", int(flash_wrapped))

        # Install the same guarded recurrent-layer fusion as the resident Qwen
        # build. All non-routed weights are hydrated before this wrapper is
        # attached, so it reuses the package-owned modules without changing the
        # loading contract.
        from moespresso.runtime.qwen.gdn_decode import install_fused_gdn_decode

        gdn_wrapped = install_fused_gdn_decode(model)
        object.__setattr__(model, "_moespresso_ssd_gdn_fused_layers", int(gdn_wrapped))
        model.eval()
        tokenizer = _load_qwen_streaming_tokenizer(
            package_dir,
            cfg,
            load_tokenizer_fn=load_tokenizer,
        )
        object.__setattr__(model, "_moespresso_ssd_streaming_lock", run_lock)
        object.__setattr__(model, "_moespresso_ssd_streaming_capacity", int(capacity_per_layer))
        object.__setattr__(
            model,
            "_moespresso_ssd_streaming_capacity_overrides",
            {
                int(layer): int(capacity)
                for layer, capacity in dict(capacity_overrides or {}).items()
            },
        )
        object.__setattr__(
            model,
            "_moespresso_ssd_streaming_eviction_policy",
            eviction_policy,
        )
        if budget_payload is not None:
            object.__setattr__(
                model,
                "_moespresso_ssd_streaming_capacity_budget",
                budget_payload,
            )
        return model, tokenizer, installed
    except Exception:
        if run_lock is not None:
            run_lock.close()
        raise


def ssd_streaming_stats(model) -> dict:
    """Summarize pooled expert residency and miss counters."""
    modules = resident_slots = hits = misses = loads = evictions = 0
    native_publication_modules = 0
    native_publication = dict.fromkeys(
        (
            "native_calls",
            "published",
            "miss",
            "pending",
            "suppressed",
            "map_builds",
            "ineligible",
            "poll_slices",
            "poll_yields",
            "timed_out",
        ),
        0,
    )
    load_seconds = 0.0
    calls = decode_calls = prefill_calls = direct_calls = 0
    row_chunked_calls = sorted_chunked_calls = over_capacity_calls = 0
    segmented_prefill_calls = 0
    unified_sorted_prefill_calls = 0
    barrier_free_prefill_calls = 0
    barrier_free_identity_calls = 0
    barrier_free_fused_swiglu_calls = 0
    barrier_free_decode_calls = 0
    barrier_free_decode_flush_calls = 0
    decode_routed_fused_calls = 0
    pipelined_decode_fused_calls = 0
    hc_fused_pre_calls = 0
    hc_fused_post_calls = 0
    hc_fused_pre_decode_calls = 0
    hc_fused_pre_tail_decode_calls = 0
    hc_fused_post_decode_calls = 0
    projection_load_wait_calls = projection_no_miss_calls = 0
    projection_load_parallel_calls = 0
    projection_load_wait_seconds = 0.0
    overlap_load_started_calls = overlap_load_wait_calls = 0
    overlap_load_wait_seconds = overlap_load_total_seconds = 0.0
    overlap_load_hidden_seconds = overlap_shared_eval_seconds = 0.0
    overlap_shared_eval_calls = overlap_prefill_no_eval_calls = 0
    overlap_no_miss_calls = overlap_skipped_over_capacity_calls = 0
    overlap_ticket_mismatch_calls = 0
    prefetch_ticket_submitted = prefetch_ticket_consumed = 0
    prefetch_ticket_mismatched = prefetch_ticket_stale = 0
    prefetch_ticket_experts = prefetch_ticket_loaded = 0
    prefetch_ticket_wait_seconds = 0.0
    expert_spec_prefetch_loads = expert_spec_prefetch_skips = 0
    token_layers = unique_active_experts = chunks = 0
    seen_experts = prefill_seen_experts = decode_seen_experts = 0
    max_unique_active_experts = 0
    index_sync_calls = index_resync_calls = 0
    index_sync_seconds = index_resync_seconds = 0.0
    routed_build_seconds = 0.0
    decode_moe_block_calls = 0
    decode_moe_block_seconds = 0.0
    router_gate_seconds = 0.0
    router_export_seconds = 0.0
    shared_experts_build_seconds = 0.0
    block_exit_kick_seconds = 0.0
    routed_weighted_sum_calls = 0
    routed_weighted_sum_slot_elements = 0
    routed_weighted_sum_output_elements = 0
    slot_table_rebuilds = 0
    compiled_island_calls = 0
    block_exit_kick_calls = 0
    pipelined_layers = 0
    pipeline_read_seconds = pipeline_join_seconds = 0.0
    bundle_row_preads = bundle_cached_takes = 0
    bundle_row_read_bytes = 0
    routed_matmul_calls = routed_matmul_slot_elements = 0
    q6_down_qmv_calls = 0
    iqk_decode_flush_calls = 0
    iqk_verify_flush_calls = 0
    iqk_dual_gemv_calls = iqk_dual_gemv_pairs = 0
    iqk_gemv_calls = iqk_gemv_pairs = 0
    iqk_two_dispatch_calls = iqk_two_dispatch_routes = 0
    iqk_bounded_counters = dict.fromkeys((
        "iqk_bounded_two_dispatch_calls",
        "iqk_bounded_two_dispatch_routes",
        "iqk_bounded_two_dispatch_gate_up_slot_mismatch_fallbacks",
    ), 0)
    iqk_sorted_prefill_calls = iqk_sorted_prefill_pairs = 0
    iqk_packed_prefill_calls = iqk_packed_prefill_pairs = 0
    iqk_sorted_nsplit_calls = iqk_sorted_nsplit_parts = 0
    routed_projection_matmul_calls = {projection: 0 for projection in _SWITCH_PROJECTIONS}
    routed_projection_matmul_slot_elements = {projection: 0 for projection in _SWITCH_PROJECTIONS}
    for layer in _layers(model):
        hc_fused_pre_calls += int(getattr(layer, "_moespresso_dsv4_hc_fused_pre_calls", 0) or 0)
        hc_fused_post_calls += int(getattr(layer, "_moespresso_dsv4_hc_fused_post_calls", 0) or 0)
        hc_fused_pre_decode_calls += int(
            getattr(layer, "_moespresso_dsv4_hc_fused_pre_decode_calls", 0) or 0
        )
        hc_fused_pre_tail_decode_calls += int(
            getattr(layer, "_moespresso_dsv4_hc_fused_pre_tail_decode_calls", 0) or 0
        )
        hc_fused_post_decode_calls += int(
            getattr(layer, "_moespresso_dsv4_hc_fused_post_decode_calls", 0) or 0
        )
        mlp = getattr(layer, "mlp", None)
        # Block-level IQ_K decode and verify commits; zero on other codecs.
        iqk_decode_flush_calls += int(getattr(mlp, "iqk_decode_flush_calls", 0) or 0)
        iqk_verify_flush_calls += int(getattr(mlp, "iqk_verify_flush_calls", 0) or 0)
        switch = getattr(mlp, "switch_mlp", None)
        if switch is None:
            switch = _pooled_switch_for_layer(layer)
        # IQ_K routed seam counters. Resident reference switches and pooled
        # target switches expose the same fields. `nsplit_parts` is a setting,
        # so it takes the maximum.
        if type(switch).__name__ == "IqkDeepseekV4SwitchGLU" or bool(
            getattr(switch, "_all_iqk", False)
        ):
            iqk_gemv_calls += int(getattr(switch, "gemv_calls", 0) or 0)
            iqk_gemv_pairs += int(getattr(switch, "gemv_pairs", 0) or 0)
            iqk_sorted_prefill_calls += int(getattr(switch, "sorted_prefill_calls", 0) or 0)
            iqk_sorted_prefill_pairs += int(getattr(switch, "sorted_prefill_pairs", 0) or 0)
            iqk_packed_prefill_calls += int(getattr(switch, "packed_prefill_calls", 0) or 0)
            iqk_packed_prefill_pairs += int(getattr(switch, "packed_prefill_pairs", 0) or 0)
            iqk_sorted_nsplit_calls += int(getattr(switch, "sorted_nsplit_calls", 0) or 0)
            iqk_sorted_nsplit_parts = max(
                iqk_sorted_nsplit_parts, int(getattr(switch, "sorted_nsplit_parts", 0) or 0)
            )
            iqk_decode_flush_calls += int(getattr(switch, "iqk_decode_flush_calls", 0) or 0)
            iqk_verify_flush_calls += int(getattr(switch, "iqk_verify_flush_calls", 0) or 0)
            iqk_dual_gemv_calls += int(getattr(switch, "iqk_dual_gemv_calls", 0) or 0)
            iqk_dual_gemv_pairs += int(getattr(switch, "iqk_dual_gemv_pairs", 0) or 0)
            iqk_two_dispatch_calls += int(getattr(switch, "iqk_two_dispatch_calls", 0) or 0)
            iqk_two_dispatch_routes += int(getattr(switch, "iqk_two_dispatch_routes", 0) or 0)
            for name in iqk_bounded_counters:
                iqk_bounded_counters[name] += int(getattr(switch, name, 0) or 0)
        if not isinstance(switch, PooledSwitchGLU):
            continue
        modules += 1
        publication = getattr(switch, "_qwen4_native_publication", None)
        if publication is not None:
            publication_stats = publication.snapshot()
            native_publication_modules += 1
            for name in native_publication:
                native_publication[name] += int(publication_stats.get(name, 0) or 0)
        # the layer's three pools share one BundleRowCache; count it once
        row_cache = switch.gate_proj.pool.row_cache
        if row_cache is not None:
            bundle_row_preads += row_cache.total_preads
            bundle_cached_takes += row_cache.total_cached_takes
            bundle_row_read_bytes += row_cache.total_preads * row_cache.index.row_bytes(layer=row_cache.layer)
        calls += switch.total_calls
        decode_calls += switch.decode_calls
        prefill_calls += switch.prefill_calls
        direct_calls += switch.direct_calls
        row_chunked_calls += switch.row_chunked_calls
        sorted_chunked_calls += switch.sorted_chunked_calls
        segmented_prefill_calls += switch.segmented_prefill_calls
        unified_sorted_prefill_calls += getattr(switch, "unified_sorted_prefill_calls", 0)
        barrier_free_prefill_calls += getattr(switch, "barrier_free_prefill_calls", 0)
        barrier_free_identity_calls += getattr(switch, "barrier_free_identity_calls", 0)
        barrier_free_fused_swiglu_calls += getattr(switch, "barrier_free_fused_swiglu_calls", 0)
        barrier_free_decode_calls += getattr(switch, "barrier_free_decode_calls", 0)
        barrier_free_decode_flush_calls += getattr(switch, "barrier_free_decode_flush_calls", 0)
        decode_routed_fused_calls += getattr(switch, "decode_routed_fused_calls", 0)
        pipelined_decode_fused_calls += getattr(switch, "pipelined_decode_fused_calls", 0)
        over_capacity_calls += switch.over_capacity_calls
        projection_load_wait_calls += switch.projection_load_wait_calls
        projection_no_miss_calls += switch.projection_no_miss_calls
        projection_load_parallel_calls += switch.projection_load_parallel_calls
        projection_load_wait_seconds += switch.projection_load_wait_seconds
        overlap_load_started_calls += switch.overlap_load_started_calls
        overlap_load_wait_calls += switch.overlap_load_wait_calls
        overlap_load_wait_seconds += switch.overlap_load_wait_seconds
        overlap_load_total_seconds += switch.overlap_load_total_seconds
        overlap_load_hidden_seconds += switch.overlap_load_hidden_seconds
        overlap_shared_eval_calls += switch.overlap_shared_eval_calls
        overlap_shared_eval_seconds += switch.overlap_shared_eval_seconds
        overlap_prefill_no_eval_calls += switch.overlap_prefill_no_eval_calls
        overlap_no_miss_calls += switch.overlap_no_miss_calls
        overlap_skipped_over_capacity_calls += switch.overlap_skipped_over_capacity_calls
        overlap_ticket_mismatch_calls += switch.overlap_ticket_mismatch_calls
        prefetch_ticket_submitted += switch.prefetch_ticket_submitted
        prefetch_ticket_consumed += switch.prefetch_ticket_consumed
        prefetch_ticket_mismatched += switch.prefetch_ticket_mismatched
        prefetch_ticket_stale += switch.prefetch_ticket_stale
        prefetch_ticket_experts += switch.prefetch_ticket_experts
        prefetch_ticket_loaded += switch.prefetch_ticket_loaded
        prefetch_ticket_wait_seconds += switch.prefetch_ticket_wait_seconds
        token_layers += switch.total_token_layers
        unique_active_experts += switch.total_unique_active_experts
        seen_experts += len(switch.seen_experts)
        prefill_seen_experts += len(switch.prefill_seen_experts)
        decode_seen_experts += len(switch.decode_seen_experts)
        max_unique_active_experts = max(
            max_unique_active_experts,
            switch.max_unique_active_experts,
        )
        chunks += switch.total_chunks
        index_sync_calls += switch.index_sync_calls
        index_sync_seconds += switch.index_sync_seconds
        index_resync_calls += switch.index_resync_calls
        index_resync_seconds += switch.index_resync_seconds
        routed_build_seconds += switch.routed_build_seconds
        decode_moe_block_calls += switch.decode_moe_block_calls
        decode_moe_block_seconds += switch.decode_moe_block_seconds
        router_gate_seconds += switch.router_gate_seconds
        router_export_seconds += switch.router_export_seconds
        shared_experts_build_seconds += switch.shared_experts_build_seconds
        block_exit_kick_seconds += switch.block_exit_kick_seconds
        routed_weighted_sum_calls += switch.routed_weighted_sum_calls
        routed_weighted_sum_slot_elements += switch.routed_weighted_sum_slot_elements
        routed_weighted_sum_output_elements += switch.routed_weighted_sum_output_elements
        compiled_island_calls += switch.compiled_island_calls
        block_exit_kick_calls += switch.block_exit_kick_calls
        pipelined_layers += switch.pipelined_layers
        pipeline_read_seconds += switch.pipeline_read_seconds
        pipeline_join_seconds += switch.pipeline_join_seconds
        for pool in _unique_projection_pools_for_switch(switch):
            resident_slots += len(pool.resident_ids())
            hits += pool.total_hits
            misses += pool.total_misses
            loads += pool.total_loads
            evictions += pool.total_evictions
            load_seconds += pool.total_load_seconds
            slot_table_rebuilds += pool.slot_table_rebuilds
            expert_spec_prefetch_loads += pool.total_prefetch_loads
            expert_spec_prefetch_skips += pool.total_prefetch_skips
        for projection, module in _projection_modules_for_switch(switch):
            matmul_calls = int(getattr(module, "matmul_slot_calls", 0))
            matmul_elements = int(getattr(module, "matmul_slot_elements", 0))
            routed_matmul_calls += matmul_calls
            routed_matmul_slot_elements += matmul_elements
            routed_projection_matmul_calls[projection] += matmul_calls
            routed_projection_matmul_slot_elements[projection] += matmul_elements
            q6_down_qmv_calls += int(getattr(module, "decode_q6_qmv_calls", 0))

    # The DS4 ratio-4 prefill consumer is selected in the kernel module, so
    # its engagement counts are module totals rather than per-layer attributes.
    # The tiled score operand and grouped wo_a projection use the same scheme.
    from moespresso.runtime.deepseek_v4.indexed_attention_kernel import (
        indexer_scores_call_counts,
        prefill_consumer_call_counts,
    )
    from moespresso.runtime.deepseek_v4.model import (
        affine_wo_fp32_call_counts,
        attention_seam_rope_call_counts,
        banded_prefill_call_counts,
        kquant_bulk_route_call_counts,
        q8_dense_matmul_call_counts,
        q8_ffn_hc_post_call_counts,
        q8_hc_post_call_counts,
        router_gate_trim_call_counts,
        wo_a_projection_call_counts,
    )

    # `iqk_engagement` reports built dequant-range kernel keys. The counters
    # expose the registry size so phase-level callers can identify new builds.
    from mlx_iqk.kernels import built_dequant_range_kernels
    from moespresso.runtime.deepseek_v4.iqk_decode_kernel import (
        built_dual_gemv_kernels,
    )

    consumer_counts = prefill_consumer_call_counts()
    scores_counts = indexer_scores_call_counts()
    wo_a_counts = wo_a_projection_call_counts()
    banded_counts = banded_prefill_call_counts()
    seam_rope_counts = attention_seam_rope_call_counts()
    router_trim_counts = router_gate_trim_call_counts()
    q8_dense_counts = q8_dense_matmul_call_counts()
    q8_hc_post_counts = q8_hc_post_call_counts()
    q8_ffn_hc_post_counts = q8_ffn_hc_post_call_counts()
    affine_wo_counts = affine_wo_fp32_call_counts()
    kquant_bulk_counts = kquant_bulk_route_call_counts()

    # Flash D=256 prefill engagement, the same route the resident build installs;
    # the streamed build wraps the identical `self_attn` modules. Reachable on
    # the streamed model whether or not an eligible wrapper is present.
    from moespresso.runtime.qwen.full_attention import (
        flash_prefill_attention_stats,
    )

    flash_counts = flash_prefill_attention_stats(model)

    from moespresso.runtime.qwen.gdn_decode import fused_gdn_decode_stats

    gdn_counts = fused_gdn_decode_stats(model)

    from moespresso.runtime.qwen.router_gemv import router_bf16_f32_stats

    router_gemv_counts = router_bf16_f32_stats(model)

    # Record the load-time drafter decision with the engagement counters. The
    # same payload is exported through each runtime statistics surface.
    drafter_policy = getattr(model, "_moespresso_ds4_drafter_policy", None)
    policy_mode = (drafter_policy or {}).get("mode")
    policy_decision = (drafter_policy or {}).get("decision")
    pooled_session = getattr(model, "_moespresso_pooled_decode_session", None)
    bound_gate = getattr(pooled_session, "_gate_mod", None)

    from moespresso.runtime.native_gate import gate_is_loaded

    total = hits + misses
    routing_stats = getattr(model, "cache_routing_stats", None)
    return {
        "enabled": modules > 0,
        **({"cache_routing": routing_stats()} if callable(routing_stats) else {}),
        "shared_pooled_decode": pooled_session is not None,
        "native_gate_bound": bound_gate is not None and bound_gate is not False,
        "native_gate_loaded": gate_is_loaded(),
        "qwen_native_publication_modules": native_publication_modules,
        **{
            "qwen_native_publication_" + name: value
            for name, value in native_publication.items()
        },
        "kquant_dense_modules_installed": int(
            getattr(model, "_moespresso_ssd_kquant_dense_installed", 0) or 0
        ),
        "flash_prefill_wrapped_layers": flash_counts["wrapped_layers"],
        "flash_prefill_calls": flash_counts["flash_calls"],
        "q8_decode_tile16_calls": flash_counts["decode_calls"],
        "q8_decode_dimension_merge_calls": flash_counts["decode_dimension_merge_calls"],
        "flash_prefill_fallback_no_cache": flash_counts["fallback_no_cache"],
        "flash_prefill_fallback_decode": flash_counts["fallback_decode"],
        "flash_prefill_fallback_cache": flash_counts["fallback_cache"],
        "flash_prefill_fallback_mask": flash_counts["fallback_mask"],
        "flash_prefill_fallback_geometry": flash_counts["fallback_geometry"],
        "flash_prefill_fallback_dtype": flash_counts["fallback_dtype"],
        "flash_prefill_fallback_kernel": flash_counts["fallback_kernel"],
        "gdn_fused_wrapped_layers": gdn_counts["wrapped_layers"],
        "gdn_fused_calls": gdn_counts["fused_calls"],
        "gdn_fused_rms_scale_calls": gdn_counts["rms_scale_fused_calls"],
        "gdn_fused_fallback_training": gdn_counts["fallback_training"],
        "gdn_fused_fallback_input": gdn_counts["fallback_input"],
        "gdn_fused_fallback_mask": gdn_counts["fallback_mask"],
        "gdn_fused_fallback_cache": gdn_counts["fallback_cache"],
        "gdn_fused_fallback_geometry": gdn_counts["fallback_geometry"],
        "gdn_fused_fallback_dtype": gdn_counts["fallback_dtype"],
        "gdn_fused_fallback_kernel": gdn_counts["fallback_kernel"],
        "gdn_fused_fallback_qkv": gdn_counts["fallback_qkv"],
        "router_bf16_f32_wrapped_layers": router_gemv_counts["wrapped_layers"],
        "router_bf16_f32_validated_layers": router_gemv_counts["validated_layers"],
        "router_bf16_f32_kernel_calls": router_gemv_counts["kernel_calls"],
        "router_bf16_f32_fallback_training": router_gemv_counts["fallback_training"],
        "router_bf16_f32_fallback_input_shape": router_gemv_counts["fallback_input_shape"],
        "router_bf16_f32_fallback_input_dtype": router_gemv_counts["fallback_input_dtype"],
        "router_bf16_f32_fallback_weight_contract": router_gemv_counts["fallback_weight_contract"],
        "router_bf16_f32_fallback_kernel": router_gemv_counts["fallback_kernel"],
        "r4_prefill_consumer_mma_calls": consumer_counts["mma"],
        "r4_prefill_consumer_scalar_calls": consumer_counts["v1"],
        "r4_prefill_scores_f16_calls": scores_counts["f16"],
        "r4_prefill_scores_f32_calls": scores_counts["f32"],
        "wo_a_batched_decode_calls": wo_a_counts["batched_decode"],
        "wo_a_batched_tiny_m_calls": wo_a_counts["batched_tiny_m"],
        "wo_a_gather_decode_calls": wo_a_counts["gather_decode"],
        "wo_a_loop_projection_calls": wo_a_counts["loop"],
        "q8_dense_decode_qmv_calls": q8_dense_counts["decode_qmv"],
        "q8_dense_decode_wire_qmv_wo_b_calls": (q8_dense_counts["decode_wire_qmv_wo_b"]),
        "q8_dense_decode_wire_qmv_lm_head_calls": (q8_dense_counts["decode_wire_qmv_lm_head"]),
        "q8_dense_tiny_m_qmm_wo_b_calls": (q8_dense_counts["tiny_m_qmm_wo_b"]),
        "q8_dense_prefill_dequant_calls": q8_dense_counts["prefill_dequant"],
        "kquant_bulk_kernel_calls": kquant_bulk_counts["kernel"],
        "kquant_bulk_bridge_calls": kquant_bulk_counts["bridge"],
        "q8_hc_post_engaged_calls": q8_hc_post_counts["engaged"],
        "q8_hc_post_fallback_calls": q8_hc_post_counts["fallback"],
        "q8_hc_post_delegated_calls": q8_hc_post_counts["delegated"],
        "q8_ffn_hc_post_engaged_calls": q8_ffn_hc_post_counts["engaged"],
        "q8_ffn_hc_post_fallback_calls": q8_ffn_hc_post_counts["fallback"],
        "q8_ffn_hc_post_delegated_calls": q8_ffn_hc_post_counts["delegated"],
        "affine_wo_fp32_wo_a_calls": affine_wo_counts["wo_a"],
        "affine_wo_fp32_wo_b_calls": affine_wo_counts["wo_b"],
        "banded_prefill_mma_calls": banded_counts["mma"],
        "banded_prefill_sdpa_calls": banded_counts["sdpa"],
        "banded_prefill_mma_offset_calls": banded_counts["mma_offset"],
        "banded_prefill_composed_offset_calls": banded_counts["composed_offset"],
        "attn_seam_rope_fused_calls": seam_rope_counts["fused"],
        "attn_seam_rope_composed_calls": seam_rope_counts["composed"],
        "router_gate_precast_calls": router_trim_counts["precast"],
        "router_gate_select_kernel_calls": router_trim_counts["select_kernel"],
        "router_gate_select_composed_calls": (router_trim_counts["select_composed"]),
        "router_gate_composed_calls": router_trim_counts["composed"],
        "switch_modules": modules,
        "resident_slots": resident_slots,
        "expert_hits": hits,
        "expert_misses": misses,
        "expert_loads": loads,
        "expert_evictions": evictions,
        "expert_load_seconds": load_seconds,
        # A miss reads one bundle row shared by the layer's three pools;
        # cached_takes counts deduplicated row-cache reads.
        "bundle_row_preads": bundle_row_preads,
        "bundle_cached_takes": bundle_cached_takes,
        "bundle_row_read_bytes": bundle_row_read_bytes,
        "hit_rate": hits / total if total else 0.0,
        "switch_calls": calls,
        "decode_calls": decode_calls,
        "prefill_calls": prefill_calls,
        "direct_calls": direct_calls,
        "row_chunked_calls": row_chunked_calls,
        "sorted_chunked_calls": sorted_chunked_calls,
        "segmented_prefill_calls": segmented_prefill_calls,
        "unified_sorted_prefill_calls": unified_sorted_prefill_calls,
        "barrier_free_prefill_calls": barrier_free_prefill_calls,
        "barrier_free_identity_calls": barrier_free_identity_calls,
        "barrier_free_fused_swiglu_calls": barrier_free_fused_swiglu_calls,
        "barrier_free_decode_calls": barrier_free_decode_calls,
        "barrier_free_decode_flush_calls": barrier_free_decode_flush_calls,
        "decode_routed_fused_calls": decode_routed_fused_calls,
        "pipelined_decode_fused_calls": pipelined_decode_fused_calls,
        "iqk_decode_flush_calls": iqk_decode_flush_calls,
        "iqk_verify_flush_calls": iqk_verify_flush_calls,
        "iqk_dual_gemv_calls": iqk_dual_gemv_calls,
        "iqk_dual_gemv_pairs": iqk_dual_gemv_pairs,
        "iqk_gemv_calls": iqk_gemv_calls,
        "iqk_gemv_pairs": iqk_gemv_pairs,
        "iqk_two_dispatch_calls": iqk_two_dispatch_calls,
        "iqk_two_dispatch_routes": iqk_two_dispatch_routes,
        **iqk_bounded_counters,
        "iqk_sorted_prefill_calls": iqk_sorted_prefill_calls,
        "iqk_sorted_prefill_pairs": iqk_sorted_prefill_pairs,
        "iqk_packed_prefill_calls": iqk_packed_prefill_calls,
        "iqk_packed_prefill_pairs": iqk_packed_prefill_pairs,
        "iqk_sorted_nsplit_calls": iqk_sorted_nsplit_calls,
        "iqk_sorted_nsplit_parts": iqk_sorted_nsplit_parts,
        "built_dequant_range_kernel_count": len(built_dequant_range_kernels()),
        "built_iqk_dual_gemv_kernel_count": len(built_dual_gemv_kernels()),
        "drafter_policy": drafter_policy,
        "ds4_drafter_policy_auto_on": int(policy_mode == "auto" and policy_decision == "on"),
        "ds4_drafter_policy_auto_off": int(policy_mode == "auto" and policy_decision == "off"),
        "ds4_drafter_policy_override": int(policy_mode == "override"),
        "hc_fused_pre_calls": hc_fused_pre_calls,
        "hc_fused_post_calls": hc_fused_post_calls,
        "hc_fused_pre_decode_calls": hc_fused_pre_decode_calls,
        "hc_fused_pre_tail_decode_calls": hc_fused_pre_tail_decode_calls,
        "hc_fused_post_decode_calls": hc_fused_post_decode_calls,
        "over_capacity_calls": over_capacity_calls,
        "projection_load_wait_calls": projection_load_wait_calls,
        "projection_no_miss_calls": projection_no_miss_calls,
        "projection_load_parallel_calls": projection_load_parallel_calls,
        "projection_load_wait_seconds": projection_load_wait_seconds,
        "overlap_load_started_calls": overlap_load_started_calls,
        "overlap_load_wait_calls": overlap_load_wait_calls,
        "overlap_load_wait_seconds": overlap_load_wait_seconds,
        "overlap_load_total_seconds": overlap_load_total_seconds,
        "overlap_load_hidden_seconds": overlap_load_hidden_seconds,
        "overlap_shared_eval_calls": overlap_shared_eval_calls,
        "overlap_shared_eval_seconds": overlap_shared_eval_seconds,
        "overlap_prefill_no_eval_calls": overlap_prefill_no_eval_calls,
        "overlap_no_miss_calls": overlap_no_miss_calls,
        "overlap_skipped_over_capacity_calls": overlap_skipped_over_capacity_calls,
        "overlap_ticket_mismatch_calls": overlap_ticket_mismatch_calls,
        "prefetch_ticket_submitted": prefetch_ticket_submitted,
        "prefetch_ticket_consumed": prefetch_ticket_consumed,
        "prefetch_ticket_mismatched": prefetch_ticket_mismatched,
        "prefetch_ticket_stale": prefetch_ticket_stale,
        "prefetch_ticket_experts": prefetch_ticket_experts,
        "prefetch_ticket_loaded": prefetch_ticket_loaded,
        "prefetch_ticket_wait_seconds": prefetch_ticket_wait_seconds,
        "expert_spec_prefetch_loads": expert_spec_prefetch_loads,
        "expert_spec_prefetch_skips": expert_spec_prefetch_skips,
        "index_sync_calls": index_sync_calls,
        "index_sync_seconds": index_sync_seconds,
        "index_resync_calls": index_resync_calls,
        "index_resync_seconds": index_resync_seconds,
        "routed_build_seconds": routed_build_seconds,
        "decode_moe_block_calls": decode_moe_block_calls,
        "decode_moe_block_seconds": decode_moe_block_seconds,
        "router_gate_seconds": router_gate_seconds,
        "router_export_seconds": router_export_seconds,
        "shared_experts_build_seconds": shared_experts_build_seconds,
        "block_exit_kick_seconds": block_exit_kick_seconds,
        "routed_weighted_sum_calls": routed_weighted_sum_calls,
        "routed_weighted_sum_slot_elements": routed_weighted_sum_slot_elements,
        "routed_weighted_sum_output_elements": routed_weighted_sum_output_elements,
        "slot_table_rebuilds": slot_table_rebuilds,
        "compiled_island_calls": compiled_island_calls,
        "block_exit_kick_calls": block_exit_kick_calls,
        "routed_matmul_calls": routed_matmul_calls,
        "routed_matmul_slot_elements": routed_matmul_slot_elements,
        "routed_gate_matmul_calls": routed_projection_matmul_calls["gate_proj"],
        "routed_up_matmul_calls": routed_projection_matmul_calls["up_proj"],
        "routed_down_matmul_calls": routed_projection_matmul_calls["down_proj"],
        "q6_down_qmv_calls": q6_down_qmv_calls,
        "routed_gate_matmul_slot_elements": (routed_projection_matmul_slot_elements["gate_proj"]),
        "routed_up_matmul_slot_elements": (routed_projection_matmul_slot_elements["up_proj"]),
        "routed_down_matmul_slot_elements": (routed_projection_matmul_slot_elements["down_proj"]),
        "pipelined_layers": pipelined_layers,
        "pipeline_read_seconds": pipeline_read_seconds,
        "pipeline_join_seconds": pipeline_join_seconds,
        "token_layers": token_layers,
        "unique_active_experts": unique_active_experts,
        "seen_experts": seen_experts,
        "prefill_seen_experts": prefill_seen_experts,
        "decode_seen_experts": decode_seen_experts,
        "max_unique_active_experts": max_unique_active_experts,
        "chunk_count": chunks,
        "capacity_per_layer": getattr(model, "_moespresso_ssd_streaming_capacity", None),
        "capacity_overrides": getattr(model, "_moespresso_ssd_streaming_capacity_overrides", {}),
        "eviction_policy": getattr(model, "_moespresso_ssd_streaming_eviction_policy", None),
        "capacity_budget": getattr(model, "_moespresso_ssd_streaming_capacity_budget", None),
        "adaptive_growth": getattr(model, "_moespresso_ssd_streaming_adaptive_growth", None),
    }


def suggest_capacity_overrides_from_layer_stats(
    rows: list[dict],
    *,
    extra_slot_budget: int | None = None,
    extra_byte_budget: int | None = None,
    replacement_headroom_bytes: int | None = None,
    target: str = "all",
) -> dict[int, int]:
    """Greedily spend extra residency budget where observed churn is highest.

    Byte-budget planning also reserves enough live headroom for the complete
    detached replacement of each layer. Once a layer commits, its old storage
    can back the next transaction, so only the capacity delta remains charged
    against subsequent replacement headroom.
    """
    if (extra_slot_budget is None) == (extra_byte_budget is None):
        raise ValueError("pass exactly one of extra_slot_budget or extra_byte_budget")
    if extra_byte_budget is not None and replacement_headroom_bytes is None:
        raise ValueError("extra_byte_budget requires replacement_headroom_bytes")
    if extra_slot_budget is not None and replacement_headroom_bytes is not None:
        raise ValueError("replacement_headroom_bytes is only valid with extra_byte_budget")
    if target not in {"all", "decode"}:
        raise ValueError("target must be 'all' or 'decode'")

    remaining_slots = int(extra_slot_budget) if extra_slot_budget is not None else None
    remaining_bytes = int(extra_byte_budget) if extra_byte_budget is not None else None
    remaining_replacement_headroom = (
        int(replacement_headroom_bytes) if replacement_headroom_bytes is not None else None
    )
    if remaining_slots is not None and remaining_slots <= 0:
        return {}
    if remaining_bytes is not None and remaining_bytes <= 0:
        return {}

    def _row_cost(row: dict) -> int:
        cost = int(row.get("slot_bytes", 0))
        if cost <= 0:
            raise ValueError("extra_byte_budget requires positive slot_bytes in every row")
        return cost

    candidates = []
    for row in rows:
        current = int(row.get("capacity", 0))
        target_capacity = (
            int(row.get("decode_seen_experts", 0))
            if target == "decode"
            else max(
                int(row.get("seen_experts", 0)),
                int(row.get("decode_seen_experts", 0)),
                int(row.get("max_unique_active_experts", 0)),
            )
        )
        max_capacity = int(
            row.get(
                "max_capacity",
                row.get("num_experts", target_capacity),
            )
        )
        target_capacity = min(target_capacity, max_capacity)
        need = target_capacity - current
        if need <= 0:
            continue
        candidates.append(
            (
                int(row.get("expert_loads", 0)),
                int(row.get("expert_misses", 0)),
                target_capacity,
                int(row["layer"]),
                current,
                need,
                _row_cost(row) if remaining_bytes is not None else 1,
                max(0, int(row.get("spare_slots", 0))),
            )
        )

    overrides: dict[int, int] = {}
    for (
        _loads,
        _misses,
        _target,
        layer,
        current,
        need,
        slot_bytes,
        spare_slots,
    ) in sorted(candidates, reverse=True):
        if remaining_slots is not None:
            if remaining_slots <= 0:
                break
            grant = min(need, remaining_slots)
            remaining_slots -= grant
        elif remaining_bytes is not None:
            assert remaining_replacement_headroom is not None
            max_grant_for_replacement = (
                remaining_replacement_headroom // slot_bytes - spare_slots - current
            )
            if remaining_bytes < slot_bytes or max_grant_for_replacement <= 0:
                continue
            grant = min(
                need,
                remaining_bytes // slot_bytes,
                max_grant_for_replacement,
            )
            remaining_bytes -= grant * slot_bytes
            remaining_replacement_headroom -= grant * slot_bytes
        else:  # pragma: no cover - guarded above
            break
        overrides[layer] = current + grant
    return overrides


def grow_ssd_streaming_capacity(
    model,
    overrides: Mapping[int, int],
    *,
    seed_hot: bool = False,
) -> dict[int, int]:
    """Grow selected routed-layer pools and return applied capacities."""
    requested = {int(layer): int(capacity) for layer, capacity in overrides.items()}
    applied: dict[int, int] = {}
    current_overrides = dict(
        getattr(
            model,
            "_moespresso_ssd_streaming_capacity_overrides",
            {},
        )
    )
    resolved_capacities = getattr(
        model,
        "_moespresso_ssd_streaming_resolved_capacities",
        None,
    )
    if resolved_capacities is not None:
        resolved_capacities = dict(resolved_capacities)
    layers = _layers(model)
    # Preserve the planner's insertion order. Replacement-headroom accounting
    # is sequential because each committed layer releases its old allocation
    # for reuse by the next transaction.
    for layer_idx, capacity in requested.items():
        if layer_idx < 0 or layer_idx >= len(layers):
            continue
        layer = layers[layer_idx]
        switch = _pooled_switch_for_layer(layer)
        if not isinstance(switch, PooledSwitchGLU):
            continue
        current = min(pool.capacity for pool in _unique_projection_pools_for_switch(switch))
        if capacity <= current:
            continue
        switch.grow_capacity(capacity)
        # The switch transaction has committed. Record it before optional
        # seeding or a later layer can fail, so runtime metadata never reports
        # the old capacity over already-published storage.
        applied[layer_idx] = capacity
        current_overrides[layer_idx] = capacity
        object.__setattr__(
            model,
            "_moespresso_ssd_streaming_capacity_overrides",
            dict(current_overrides),
        )
        if resolved_capacities is not None:
            resolved_capacities[layer_idx] = capacity
            object.__setattr__(
                model,
                "_moespresso_ssd_streaming_resolved_capacities",
                dict(sorted(resolved_capacities.items())),
            )
        if seed_hot:
            switch.seed_hot_free_slots()
    return applied


def _adaptive_extra_bytes(model, rows: list[dict]) -> int:
    base_capacity = int(getattr(model, "_moespresso_ssd_streaming_capacity", 0) or 0)
    if base_capacity <= 0:
        return 0
    total = 0
    for row in rows:
        extra = int(row["capacity"]) - base_capacity
        if extra > 0:
            total += extra * int(row["slot_bytes"])
    return total


def _growth_max_extra_bytes_default() -> int:
    """Total adaptive-growth budget (bytes), env-tunable.

    Growth is disabled by default so the startup planner remains the owner of
    automatic pool residency. The live-memory floor protects an explicitly
    enabled growth transaction, but it does not enforce the startup planner's
    resolved ceiling or the reported wired-memory budget.
    MOESPRESSO_SSD_GROWTH_MAX_EXTRA_GB explicitly enables a cumulative growth
    allowance above startup capacity.
    """
    import os

    raw = os.environ.get("MOESPRESSO_SSD_GROWTH_MAX_EXTRA_GB")
    if raw is not None:
        return int(float(raw) * (1 << 30))
    return 0


def maybe_adapt_ssd_streaming_capacity(
    model,
    *,
    available_bytes: int | None = None,
    min_available_bytes: int = 4 << 30,
    max_extra_bytes: int | None = None,
    seed_hot: bool = True,
) -> dict:
    """Conservatively grow hot routed layers after real request evidence exists."""
    latched_failure = getattr(
        model,
        "_moespresso_ssd_streaming_growth_latched_failure",
        None,
    )
    if latched_failure is not None:
        return latched_failure
    if max_extra_bytes is None:
        max_extra_bytes = _growth_max_extra_bytes_default()
    t0 = time.perf_counter()
    try:
        rows = ssd_streaming_layer_stats(model)
    except SSDStreamingBuildError:
        return {"enabled": False, "applied": {}}
    if not rows:
        return {"enabled": False, "applied": {}}
    if available_bytes is None:
        available_bytes = available_memory_bytes()

    used_extra_bytes = _adaptive_extra_bytes(model, rows)
    remaining_extra_bytes = max(0, int(max_extra_bytes) - used_extra_bytes)
    free_above_floor = max(0, int(available_bytes) - int(min_available_bytes))
    extra_byte_budget = min(remaining_extra_bytes, free_above_floor)

    if extra_byte_budget <= 0:
        result = {
            "enabled": True,
            "available_bytes": int(available_bytes),
            "min_available_bytes": int(min_available_bytes),
            "max_extra_bytes": int(max_extra_bytes),
            "used_extra_bytes": used_extra_bytes,
            "extra_byte_budget": 0,
            "replacement_headroom_bytes": free_above_floor,
            "plan": {},
            "applied": {},
            "seed_hot": bool(seed_hot),
            "seeded_slots": 0,
            "elapsed_seconds": time.perf_counter() - t0,
        }
        object.__setattr__(
            model,
            "_moespresso_ssd_streaming_adaptive_growth",
            result,
        )
        return result

    resident_before = sum(int(row["resident_slots"]) for row in rows)
    plan = suggest_capacity_overrides_from_layer_stats(
        rows,
        extra_byte_budget=remaining_extra_bytes,
        replacement_headroom_bytes=free_above_floor,
        target="all",
    )
    try:
        applied = grow_ssd_streaming_capacity(model, plan, seed_hot=seed_hot)
        after_rows = ssd_streaming_layer_stats(model)
    except Exception as exc:
        # Adaptive residency is an optimization after generation and cache
        # publication. A failed allocation, copy, or hot seed must not turn a
        # completed response into an HTTP/SSE failure. Each successfully grown
        # layer records its override at commit, so recover that truthful prefix
        # for diagnostics while the failed switch transaction remains unchanged.
        try:
            after_rows = ssd_streaming_layer_stats(model)
        except Exception:
            after_rows = rows
        committed_overrides = dict(
            getattr(
                model,
                "_moespresso_ssd_streaming_capacity_overrides",
                {},
            )
        )
        prior_capacity = {int(row["layer"]): int(row["capacity"]) for row in rows}
        applied = {
            int(layer): int(capacity)
            for layer, capacity in plan.items()
            if int(capacity) > prior_capacity.get(int(layer), int(capacity))
            and int(committed_overrides.get(int(layer), -1)) == int(capacity)
        }
        resident_after = sum(int(row["resident_slots"]) for row in after_rows)
        result = {
            "enabled": False,
            "latched": True,
            "available_bytes": int(available_bytes),
            "min_available_bytes": int(min_available_bytes),
            "max_extra_bytes": int(max_extra_bytes),
            "used_extra_bytes": _adaptive_extra_bytes(model, after_rows),
            "extra_byte_budget": extra_byte_budget,
            "replacement_headroom_bytes": free_above_floor,
            "plan": plan,
            "applied": applied,
            "seed_hot": bool(seed_hot),
            "seeded_slots": max(0, resident_after - resident_before),
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
            "elapsed_seconds": time.perf_counter() - t0,
        }
        object.__setattr__(
            model,
            "_moespresso_ssd_streaming_adaptive_growth",
            result,
        )
        # A failed replacement allocation/copy or hot seed is unlikely to
        # become cheaper on the next request. Keep completed responses fast by
        # returning this recorded failure directly on later adaptation calls.
        object.__setattr__(
            model,
            "_moespresso_ssd_streaming_growth_latched_failure",
            result,
        )
        if not getattr(
            model,
            "_moespresso_ssd_streaming_growth_error_logged",
            False,
        ):
            print(
                "[ssd-streaming] adaptive expert-pool growth failed; "
                f"serving continues at committed capacities ({type(exc).__name__}: "
                f"{exc})",
                flush=True,
            )
            object.__setattr__(
                model,
                "_moespresso_ssd_streaming_growth_error_logged",
                True,
            )
        return result
    resident_after = sum(int(row["resident_slots"]) for row in after_rows)
    result = {
        "enabled": True,
        "available_bytes": int(available_bytes),
        "min_available_bytes": int(min_available_bytes),
        "max_extra_bytes": int(max_extra_bytes),
        "used_extra_bytes": _adaptive_extra_bytes(
            model,
            after_rows,
        ),
        "extra_byte_budget": extra_byte_budget,
        "replacement_headroom_bytes": free_above_floor,
        "plan": plan,
        "applied": applied,
        "seed_hot": bool(seed_hot),
        "seeded_slots": max(0, resident_after - resident_before),
        "elapsed_seconds": time.perf_counter() - t0,
    }
    object.__setattr__(
        model,
        "_moespresso_ssd_streaming_adaptive_growth",
        result,
    )
    return result


def _all_pools_at_full_capacity(model) -> bool:
    """True when every routed projection pool can hold its full expert set."""
    pools_seen = False
    for layer in _layers(model):
        switch = _pooled_switch_for_layer(layer)
        if not isinstance(switch, PooledSwitchGLU):
            continue
        for pool in _unique_projection_pools_for_switch(switch):
            pools_seen = True
            if pool.capacity < pool.num_experts:
                return False
    return pools_seen


def _all_pools_fully_resident(model) -> bool:
    """True when every routed projection pool currently holds every expert."""
    pools_seen = False
    for layer in _layers(model):
        switch = _pooled_switch_for_layer(layer)
        if not isinstance(switch, PooledSwitchGLU):
            continue
        for pool in _unique_projection_pools_for_switch(switch):
            pools_seen = True
            with pool._bk_lock:
                if pool.capacity < pool.num_experts:
                    return False
                if len(pool._slot_of) != pool.num_experts:
                    return False
    return pools_seen


def seed_expert_residency(model, package_dir: str | Path) -> dict:
    """Layered cold-start seeding.

    Precedence: MOESPRESSO_SSD_PREWARM_EXPERTS=all requests full prewarming,
    while ``none`` skips the full-capacity default. When every projection pool
    can hold the full expert set, the default loads all experts at build time
    (source ``all-default``). Pool residency selects the routed prefill kernel;
    prewarming makes the full-resident kernel available from the first request
    and moves its SSD reads to model build time.

    Below full capacity the package hotlist seeds free slots. Its installed
    prior is capped by ``load_expert_hotlist`` so live demand can overtake it
    within the running process."""
    import os

    from moespresso.package.hotlist import HOTLIST_NAME

    info = {
        "source": "none",
        "path": None,
        "seeded": 0,
    }
    prewarm = os.environ.get("MOESPRESSO_SSD_PREWARM_EXPERTS", "").strip().lower()
    if prewarm and prewarm != "none":
        if prewarm != "all":
            raise SSDStreamingBuildError(
                "MOESPRESSO_SSD_PREWARM_EXPERTS must be 'all' or 'none' when set"
            )
        info["source"] = "all"
        info["seeded"] = seed_all_expert_residency(model)
        return info
    # 'none' skips the full-capacity default and continues with package seeding.
    if (
        prewarm != "none"
        and _all_pools_at_full_capacity(model)
    ):
        info["source"] = "all-default"
        info["seeded"] = seed_all_expert_residency(model)
        return info
    package_dir = Path(package_dir)
    path = package_dir / HOTLIST_NAME
    if path.exists():
        info["seeded"] = load_expert_hotlist(model, path)
        info["source"] = "package"
        info["path"] = str(path)
    return info


def seed_all_expert_residency(model) -> int:
    """Load every routed expert into already-allocated full-capacity pools.

    This is a server-style residency mode: it moves cold first-request SSD reads
    into model build time. It is only legal when every projection pool can hold
    the full expert set; otherwise a partial "all" preload would silently leave
    the first request on the demand-miss path.
    """
    total_seeded = 0
    for layer_idx, layer in enumerate(_layers(model)):
        switch = _pooled_switch_for_layer(layer)
        if not isinstance(switch, PooledSwitchGLU):
            continue
        pools = list(_unique_projection_pools_for_switch(switch))
        num_experts = pools[0].num_experts
        for pool in pools:
            if pool.num_experts != num_experts:
                raise SSDStreamingBuildError(
                    f"layer {layer_idx}: projection pools disagree on expert count"
                )
            if pool.capacity < num_experts:
                raise SSDStreamingBuildError(
                    f"layer {layer_idx}: full expert prewarm requires capacity "
                    f"{num_experts}, got {pool.capacity}"
                )

        before = sum(len(pool.resident_ids()) for pool in pools)
        row_cache = pools[0].row_cache
        chunk = int(getattr(row_cache, "max_rows", 32) or 32)
        chunk = max(1, min(chunk, num_experts))
        for start in range(0, num_experts, chunk):
            experts = list(range(start, min(start + chunk, num_experts)))
            # Keep gate/up/down for a row-cache window adjacent: this preserves
            # one bundle-row pread per expert instead of one pread per
            # projection after the cache evicts earlier rows.
            for pool in pools:
                pool.ensure(experts)
        after = sum(len(pool.resident_ids()) for pool in pools)
        total_seeded += max(0, after - before)
    return total_seeded


def load_expert_hotlist(
    model, path: str | Path, *, seed: bool = True, prior_cap: int | None = 8
) -> int:
    """Warm-start residency from a hotlist: install the demand counts into
    all three pools of each layer and (by default) seed the hottest experts
    into free slots now, moving cold-start misses into build time.

    Seeding preserves the package ranking. The installed prior is then rescaled
    so no entry exceeds ``prior_cap`` and request-time demand can update the
    ranking. ``prior_cap=None`` disables rescaling.

    Returns the number of experts seeded (0 if the file is missing)."""
    path = Path(path)
    if not path.exists():
        return 0
    payload = json.loads(path.read_text())
    if payload.get("kind") != "expert_hotlist":
        raise SSDStreamingBuildError(f"{path} is not an expert_hotlist file")
    layers = payload.get("layers", {})
    seeded = 0
    for layer_idx, layer in enumerate(_layers(model)):
        switch = _pooled_switch_for_layer(layer)
        if not isinstance(switch, PooledSwitchGLU):
            continue
        freq_raw = layers.get(str(layer_idx))
        if not freq_raw:
            continue
        freq = {int(expert): int(count) for expert, count in freq_raw.items()}
        for pool in _unique_projection_pools_for_switch(switch):
            for expert, count in freq.items():
                if 0 <= expert < pool.num_experts:
                    pool._freq[expert] = pool._freq.get(expert, 0) + count
        if seed:
            seeded += switch.seed_hot_free_slots()
        if prior_cap is not None:
            for pool in _unique_projection_pools_for_switch(switch):
                top = max(pool._freq.values(), default=0)
                if top > prior_cap:
                    scale = prior_cap / top
                    pool._freq = {e: max(1, round(c * scale)) for e, c in pool._freq.items()}
    return seeded


def ssd_streaming_layer_stats(model) -> list[dict]:
    """Per-routed-layer residency and miss counters for speed diagnosis."""
    rows = []
    for layer_idx, layer in enumerate(_layers(model)):
        switch = _pooled_switch_for_layer(layer)
        if not isinstance(switch, PooledSwitchGLU):
            continue

        resident_slots = hits = misses = loads = evictions = 0
        load_seconds = 0.0
        slot_bytes = 0
        row_cache = switch.gate_proj.pool.row_cache
        bundle_row_preads = row_cache.total_preads if row_cache else 0
        bundle_cached_takes = row_cache.total_cached_takes if row_cache else 0
        routed_matmul_calls = routed_matmul_slot_elements = 0
        q6_down_qmv_calls = 0
        routed_projection_matmul_calls = {projection: 0 for projection in _SWITCH_PROJECTIONS}
        routed_projection_matmul_slot_elements = {
            projection: 0 for projection in _SWITCH_PROJECTIONS
        }
        for pool in _unique_projection_pools_for_switch(switch):
            resident_slots += len(pool.resident_ids())
            hits += pool.total_hits
            misses += pool.total_misses
            loads += pool.total_loads
            evictions += pool.total_evictions
            load_seconds += pool.total_load_seconds
            slot_bytes += pool.slot_nbytes()
        for projection, module in _projection_modules_for_switch(switch):
            matmul_calls = int(getattr(module, "matmul_slot_calls", 0))
            matmul_elements = int(getattr(module, "matmul_slot_elements", 0))
            routed_matmul_calls += matmul_calls
            routed_matmul_slot_elements += matmul_elements
            routed_projection_matmul_calls[projection] += matmul_calls
            routed_projection_matmul_slot_elements[projection] += matmul_elements
            q6_down_qmv_calls += int(getattr(module, "decode_q6_qmv_calls", 0))

        total = hits + misses
        capacity = min(pool.capacity for pool in _unique_projection_pools_for_switch(switch))
        projection_pool_count = len(_unique_projection_pools_for_switch(switch))
        num_experts = switch.gate_proj.pool.num_experts
        spare_slots = max(pool.spare_slots for pool in _unique_projection_pools_for_switch(switch))
        max_capacity = min(
            pool.num_experts - pool.spare_slots
            for pool in _unique_projection_pools_for_switch(switch)
        )
        rows.append(
            {
                "layer": layer_idx,
                "capacity": capacity,
                "num_experts": num_experts,
                "spare_slots": spare_slots,
                "max_capacity": max_capacity,
                "projection_pool_count": projection_pool_count,
                "slot_bytes": slot_bytes,
                "resident_slots": resident_slots,
                "expert_hits": hits,
                "expert_misses": misses,
                "expert_loads": loads,
                "expert_evictions": evictions,
                "expert_load_seconds": load_seconds,
                "bundle_row_preads": bundle_row_preads,
                "bundle_cached_takes": bundle_cached_takes,
                "hit_rate": hits / total if total else 0.0,
                "switch_calls": switch.total_calls,
                "decode_calls": switch.decode_calls,
                "prefill_calls": switch.prefill_calls,
                "direct_calls": switch.direct_calls,
                "row_chunked_calls": switch.row_chunked_calls,
                "sorted_chunked_calls": switch.sorted_chunked_calls,
                "segmented_prefill_calls": switch.segmented_prefill_calls,
                "unified_sorted_prefill_calls": getattr(switch, "unified_sorted_prefill_calls", 0),
                "barrier_free_prefill_calls": getattr(switch, "barrier_free_prefill_calls", 0),
                "barrier_free_identity_calls": getattr(switch, "barrier_free_identity_calls", 0),
                "barrier_free_fused_swiglu_calls": getattr(
                    switch, "barrier_free_fused_swiglu_calls", 0
                ),
                "barrier_free_decode_calls": getattr(switch, "barrier_free_decode_calls", 0),
                "barrier_free_decode_flush_calls": getattr(
                    switch, "barrier_free_decode_flush_calls", 0
                ),
                "decode_routed_fused_calls": getattr(switch, "decode_routed_fused_calls", 0),
                "pipelined_decode_fused_calls": getattr(switch, "pipelined_decode_fused_calls", 0),
                "iqk_dual_gemv_calls": getattr(switch, "iqk_dual_gemv_calls", 0),
                "iqk_dual_gemv_pairs": getattr(switch, "iqk_dual_gemv_pairs", 0),
                "iqk_two_dispatch_calls": getattr(switch, "iqk_two_dispatch_calls", 0),
                "iqk_two_dispatch_routes": getattr(switch, "iqk_two_dispatch_routes", 0),
                "over_capacity_calls": switch.over_capacity_calls,
                "projection_load_wait_calls": switch.projection_load_wait_calls,
                "projection_no_miss_calls": switch.projection_no_miss_calls,
                "projection_load_parallel_calls": switch.projection_load_parallel_calls,
                "projection_load_wait_seconds": switch.projection_load_wait_seconds,
                "index_sync_calls": switch.index_sync_calls,
                "index_sync_seconds": switch.index_sync_seconds,
                "index_resync_calls": switch.index_resync_calls,
                "index_resync_seconds": switch.index_resync_seconds,
                "overlap_load_started_calls": switch.overlap_load_started_calls,
                "overlap_load_wait_calls": switch.overlap_load_wait_calls,
                "overlap_load_wait_seconds": switch.overlap_load_wait_seconds,
                "overlap_load_total_seconds": switch.overlap_load_total_seconds,
                "overlap_load_hidden_seconds": switch.overlap_load_hidden_seconds,
                "overlap_shared_eval_calls": switch.overlap_shared_eval_calls,
                "overlap_shared_eval_seconds": switch.overlap_shared_eval_seconds,
                "overlap_prefill_no_eval_calls": switch.overlap_prefill_no_eval_calls,
                "overlap_no_miss_calls": switch.overlap_no_miss_calls,
                "overlap_skipped_over_capacity_calls": (switch.overlap_skipped_over_capacity_calls),
                "overlap_ticket_mismatch_calls": switch.overlap_ticket_mismatch_calls,
                "prefetch_ticket_submitted": switch.prefetch_ticket_submitted,
                "prefetch_ticket_consumed": switch.prefetch_ticket_consumed,
                "prefetch_ticket_mismatched": switch.prefetch_ticket_mismatched,
                "prefetch_ticket_stale": switch.prefetch_ticket_stale,
                "prefetch_ticket_experts": switch.prefetch_ticket_experts,
                "prefetch_ticket_loaded": switch.prefetch_ticket_loaded,
                "prefetch_ticket_wait_seconds": switch.prefetch_ticket_wait_seconds,
                "routed_build_seconds": switch.routed_build_seconds,
                "decode_moe_block_calls": switch.decode_moe_block_calls,
                "decode_moe_block_seconds": switch.decode_moe_block_seconds,
                "router_gate_seconds": switch.router_gate_seconds,
                "router_export_seconds": switch.router_export_seconds,
                "shared_experts_build_seconds": switch.shared_experts_build_seconds,
                "block_exit_kick_seconds": switch.block_exit_kick_seconds,
                "routed_matmul_calls": routed_matmul_calls,
                "routed_matmul_slot_elements": routed_matmul_slot_elements,
                "routed_weighted_sum_calls": switch.routed_weighted_sum_calls,
                "routed_weighted_sum_slot_elements": (switch.routed_weighted_sum_slot_elements),
                "routed_weighted_sum_output_elements": (switch.routed_weighted_sum_output_elements),
                "routed_gate_matmul_calls": routed_projection_matmul_calls["gate_proj"],
                "routed_up_matmul_calls": routed_projection_matmul_calls["up_proj"],
                "routed_down_matmul_calls": routed_projection_matmul_calls["down_proj"],
                "q6_down_qmv_calls": q6_down_qmv_calls,
                "routed_gate_matmul_slot_elements": (
                    routed_projection_matmul_slot_elements["gate_proj"]
                ),
                "routed_up_matmul_slot_elements": (
                    routed_projection_matmul_slot_elements["up_proj"]
                ),
                "routed_down_matmul_slot_elements": (
                    routed_projection_matmul_slot_elements["down_proj"]
                ),
                "token_layers": switch.total_token_layers,
                "unique_active_experts": switch.total_unique_active_experts,
                "seen_experts": len(switch.seen_experts),
                "prefill_seen_experts": len(switch.prefill_seen_experts),
                "decode_seen_experts": len(switch.decode_seen_experts),
                "max_unique_active_experts": switch.max_unique_active_experts,
                "chunk_count": switch.total_chunks,
                "compiled_island_calls": switch.compiled_island_calls,
                "block_exit_kick_calls": switch.block_exit_kick_calls,
                "pipelined_layers": switch.pipelined_layers,
                "pipeline_read_seconds": switch.pipeline_read_seconds,
                "pipeline_join_seconds": switch.pipeline_join_seconds,
            }
        )
    return rows
