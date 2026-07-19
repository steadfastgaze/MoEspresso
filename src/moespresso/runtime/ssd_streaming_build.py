"""Build the SSD-streaming MoE runtime.

This is the product-shaped builder for BRIDGE-B: instantiate the MLX model
skeleton, replace routed SwitchGLU experts with persistent SSD-backed pools, then
load only non-routed tensors resident. It deliberately does not call JANG's
resident `load_jangtq_model`, because that materializes the routed expert stacks.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path

import mlx.core as mx

from moespresso.runtime.build import _silence_false_mistral_warning
from moespresso.runtime.expert_index import ExpertIndex, build_expert_index
from moespresso.runtime.expert_slot_pool import BundleRowCache
from moespresso.runtime.pooled_switchglu import (
    PooledDeepseekV4MoEBlock,
    PooledCombinedGateUpKQuantLinear,
    PooledKQuantSwitchLinear,
    PooledMxfp4SwitchLinear,
    PooledSparseMoeBlock,
    PooledSwitchGLU,
    PooledTurboQuantSwitchLinear,
)
from moespresso.package.bundle import KQUANT_CODEC, MXFP4_CODEC, TQ_CODEC
from moespresso.runtime.streaming_capacity import (
    available_memory_bytes,
    choose_capacity,
    is_routed_expert_payload_key,
    package_capacity_budget,
    validate_min_resident_experts,
)

_SWITCH_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
class SSDStreamingBuildError(RuntimeError):
    pass


def _layers(model) -> list:
    """Return the model's decoder layers, handling wrapper and text-only shapes."""
    candidates = (
        getattr(getattr(getattr(model, "language_model", None), "model", None),
                "layers", None),
        getattr(getattr(model, "model", None), "layers", None),
        getattr(model, "layers", None),
    )
    for layers in candidates:
        if layers is not None:
            return layers
    raise SSDStreamingBuildError("could not find decoder layers on model")


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
        raise SSDStreamingBuildError(
            f"could not infer dimensions for switch_mlp.{projection}")
    return int(in_features), int(out_features)


def _tables_for_projection(*, in_features: int, bits: int, seed: int):
    """Tiny deterministic TQ tables, without allocating a full expert stack."""
    from jang_tools.turboquant.codebook import compute_codebook
    from jang_tools.turboquant.rotation import generate_random_signs

    codebook = mx.array(compute_codebook(in_features, bits), dtype=mx.float32)
    signs = mx.array(generate_random_signs(in_features, seed=seed), dtype=mx.float32)
    mx.eval(codebook, signs)
    return codebook, signs


def _pooled_projection(
    *,
    package_dir: Path,
    index: ExpertIndex,
    layer: int,
    projection: str,
    capacity_per_layer: int,
    in_features: int,
    out_features: int,
    seed: int,
    eviction_policy: str,
    row_cache=None,
    spare_slots: int = 0,
) -> PooledTurboQuantSwitchLinear | PooledMxfp4SwitchLinear | PooledKQuantSwitchLinear:
    geometry = index.geometry(layer=layer, projection=projection)
    bits = geometry.bits
    if geometry.codec == KQUANT_CODEC:
        bytes_per_block = int(geometry.bytes_per_block or 0)
        weights_per_block = int(geometry.weights_per_block or 0)
        if bytes_per_block <= 0 or weights_per_block <= 0:
            raise SSDStreamingBuildError(
                f"layer {layer} {projection}: missing K-quant geometry")
        if geometry.packed_cols % bytes_per_block:
            raise SSDStreamingBuildError(
                f"layer {layer} {projection}: K-quant bytes_per_row "
                f"{geometry.packed_cols} is not divisible by {bytes_per_block}")
        packed_in_features = geometry.packed_cols // bytes_per_block * weights_per_block
    else:
        packed_in_features = geometry.packed_cols * (32 // bits)
    if packed_in_features != in_features:
        raise SSDStreamingBuildError(
            f"layer {layer} {projection}: skeleton input dim {in_features} "
            f"!= packed geometry dim {packed_in_features}")
    if geometry.out_features != out_features:
        raise SSDStreamingBuildError(
            f"layer {layer} {projection}: skeleton output dim {out_features} "
            f"!= packed geometry dim {geometry.out_features}")

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
    if geometry.codec == TQ_CODEC:
        codebook, signs = _tables_for_projection(
            in_features=in_features,
            bits=bits,
            seed=seed,
        )
        return PooledTurboQuantSwitchLinear(
            package_dir=package_dir,
            index=index,
            layer=layer,
            projection=projection,
            capacity=capacity_per_layer,
            codebook=codebook,
            signs=signs,
            eviction_policy=eviction_policy,
            row_cache=row_cache,
            spare_slots=spare_slots,
        )
    raise SSDStreamingBuildError(
        f"layer {layer} {projection}: unsupported expert codec {geometry.codec!r}")


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
    return tuple(
        (projection, getattr(switch, projection))
        for projection in _SWITCH_PROJECTIONS
    )


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


def install_pooled_switchglus(
    model,
    *,
    package_dir: str | Path,
    index: ExpertIndex,
    capacity_per_layer: int,
    capacity_overrides: Mapping[int, int] | None = None,
    eviction_policy: str = "lfu",
    seed: int = 42,
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
    for layer_idx, layer in enumerate(_layers(model)):
        mlp = getattr(layer, "mlp", None)
        sw = getattr(mlp, "switch_mlp", None)
        if sw is None:
            if layer_idx in index.layers_indexed():
                raise SSDStreamingBuildError(
                    f"expert index has layer {layer_idx}, but model has no switch_mlp")
            continue
        if not index.has_projection(layer=layer_idx, projection="gate_proj"):
            continue

        layer_capacity = int(capacity_overrides.get(layer_idx, capacity_per_layer))
        if layer_capacity < 1:
            raise SSDStreamingBuildError(
                f"layer {layer_idx} capacity must be >= 1")
        # One bundle-row cache per layer: a missed expert is pread once (the
        # whole bundle row) and each physical pool memcpys its slices. Combined
        # K-quant gate/up has two physical consumers (combined gate/up + down);
        # non-combinable codecs keep three.
        combine_gate_up = _can_combine_gate_up_kquant(index, layer=layer_idx)
        row_cache = BundleRowCache(
            package_dir=package_dir,
            index=index,
            layer=layer_idx,
            consumers=2 if combine_gate_up else 3,
        )
        projections = {}
        if combine_gate_up:
            gate_in, gate_out = _projection_dims(sw, "gate_proj")
            up_in, up_out = _projection_dims(sw, "up_proj")
            if gate_in != up_in or gate_out != up_out:
                raise SSDStreamingBuildError(
                    f"layer {layer_idx}: cannot combine gate/up K-quant with "
                    f"different skeleton dims gate=({gate_in}, {gate_out}) "
                    f"up=({up_in}, {up_out})")
            combined = PooledCombinedGateUpKQuantLinear(
                package_dir=package_dir,
                index=index,
                layer=layer_idx,
                capacity=layer_capacity,
                eviction_policy=eviction_policy,
                row_cache=row_cache,
                spare_slots=spare_slots,
            )
            projections["gate_proj"] = combined
            projections["up_proj"] = combined.up_alias
            in_features, out_features = _projection_dims(sw, "down_proj")
            projections["down_proj"] = _pooled_projection(
                package_dir=package_dir,
                index=index,
                layer=layer_idx,
                projection="down_proj",
                capacity_per_layer=layer_capacity,
                in_features=in_features,
                out_features=out_features,
                seed=seed,
                eviction_policy=eviction_policy,
                row_cache=row_cache,
                spare_slots=spare_slots,
            )
        else:
            for projection in _SWITCH_PROJECTIONS:
                in_features, out_features = _projection_dims(sw, projection)
                projections[projection] = _pooled_projection(
                    package_dir=package_dir,
                    index=index,
                    layer=layer_idx,
                    projection=projection,
                    capacity_per_layer=layer_capacity,
                    in_features=in_features,
                    out_features=out_features,
                    seed=seed,
                    eviction_policy=eviction_policy,
                    row_cache=row_cache,
                    spare_slots=spare_slots,
                )
        setattr(mlp, "switch_mlp", PooledSwitchGLU(
            gate_proj=projections["gate_proj"],
            up_proj=projections["up_proj"],
            down_proj=projections["down_proj"],
            activation=sw.activation,
        ))
        if all(hasattr(mlp, name) for name in (
            "gate",
            "shared_expert",
            "shared_expert_gate",
        )):
            block = PooledSparseMoeBlock(mlp)
            setattr(layer, "mlp", block)
            last_block = block
        elif wrap_deepseek_v4_moe and all(hasattr(mlp, name) for name in (
            "gate",
            "shared_experts",
        )):
            block = PooledDeepseekV4MoEBlock(mlp)
            setattr(layer, "mlp", block)
            last_block = block
        installed += 1

    if last_block is not None:
        # the pipelined decode joins its worker queue at the deepest MoE layer
        last_block.pipeline_is_last = True

    expected = set(index.layers_indexed())
    if installed != len(expected):
        raise SSDStreamingBuildError(
            f"installed {installed} pooled SwitchGLU layer(s), "
            f"but expert index declares {len(expected)} layer(s)")
    return installed


def install_lookahead(model, delta: int) -> int:
    """Wire cross-layer router lookahead: layer L predicts (and
    prefetches) layer L+delta's experts using L+delta's actual router weight
    on L's input hidden. Call after non-routed weights are resident (the
    router gates must be loaded). Returns the number of layers wired."""
    if delta <= 0:
        return 0
    blocks = []
    for layer in _layers(model):
        mlp = getattr(layer, "mlp", None)
        sw = getattr(mlp, "switch_mlp", None)
        if isinstance(sw, PooledSwitchGLU):
            blocks.append((mlp, sw))
    wired = 0
    for i, (_mlp, sw) in enumerate(blocks):
        j = i + delta
        if j >= len(blocks):
            continue
        target_mlp, target_sw = blocks[j]
        g = target_mlp.gate
        if getattr(g, "hash", False):
            # Hash-routed targets (the first DS4 layers) select experts
            # from a token-id table; the hidden state is never inspected. A score
            # prediction would be noise, so the target is skipped. No
            # layer targets a hash layer at delta >= num_hash_layers.
            continue
        if hasattr(g, "scales"):  # affine-quantized router
            w = mx.dequantize(g.weight, g.scales, g.biases,
                              group_size=g.group_size, bits=g.bits)
        else:  # fp16 passthrough (our packages)
            w = g.weight
        sw.lookahead_w = w.astype(mx.float16)
        bias = getattr(g, "bias", None)
        if bias is not None:
            # The DS4 score gate adds a per-expert selection bias after
            # the monotone score transform; the block's prediction
            # scoring needs it to rank candidates the way the real
            # gate selects them.
            sw.lookahead_b = bias.astype(mx.float32)
            mx.eval(sw.lookahead_w, sw.lookahead_b)
        else:
            sw.lookahead_b = None
            mx.eval(sw.lookahead_w)
        sw.lookahead_target = target_sw
        wired += 1
    return wired


_KQUANT_DENSE_KILL_SWITCH = "MOESPRESSO_QWEN_STREAMING_KQUANT_DENSE"


def _read_manifest(package_dir: Path) -> dict | None:
    """Best-effort manifest read for the streaming builder's dense-codec check."""
    manifest_path = package_dir / "package_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        return json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        return None


def _kquant_dense_install_enabled() -> bool:
    """The kquant-dense install engages by default; the env is a kill switch.

    It only activates for manifests that carry K-quant dense tensors, so the
    default-on setting is a no-op for every affine-dense package. Setting
    ``MOESPRESSO_QWEN_STREAMING_KQUANT_DENSE=0`` restores the old behavior
    (dense leaves stay stock affine, so a K-quant dense package refuses at load
    with the MLX uint32 error). This remains a diagnostic switch and is disabled
    by default.
    """
    import os

    return os.environ.get(_KQUANT_DENSE_KILL_SWITCH, "1") != "0"


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
    if not _kquant_dense_install_enabled():
        # Diagnostic kill switch: leave the affine leaves in place so the
        # historical refuse/crash behavior is reproducible.
        print(
            "[ssd-streaming] K-quant dense install disabled by "
            f"{_KQUANT_DENSE_KILL_SWITCH}=0; the affine leaves will refuse the "
            "K-quant wire bytes at load.",
            flush=True,
        )
        return 0
    installed = install_manifest_kquant_modules(model, manifest)
    print(
        f"[ssd-streaming] installed {installed} K-quant dense module(s) "
        "before the resident load.",
        flush=True,
    )
    return installed


def _load_non_routed_resident(model, package_dir: str | Path) -> None:
    """Load every tensor except routed-expert TQ stacks, then materialize resident."""
    from jang_tools.ssm_layout import sanitize_grouped_conv1d_layout

    package_dir = Path(package_dir)
    for shard in sorted(package_dir.glob("model-*.safetensors")):
        weights = mx.load(str(shard))
        regular = {
            key: value
            for key, value in weights.items()
            if not _is_routed_expert_key(key)
        }
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


def _mxtq_seed(package_dir: Path, default: int) -> int:
    path = package_dir / "jang_config.json"
    if not path.exists():
        return default
    return int(json.loads(path.read_text()).get("mxtq_seed", default))


def _budget_payload(budget) -> dict:
    return {
        "available_bytes": budget.available_bytes,
        "resident_base_bytes": budget.resident_base_bytes,
        "runtime_resident_bytes": budget.runtime_resident_bytes,
        "kv_activation_allowance_bytes": budget.kv_activation_allowance_bytes,
        "safety_margin_bytes": budget.safety_margin_bytes,
        "bytes_per_capacity_unit": budget.bytes_per_capacity_unit,
        "usable_bytes": budget.usable_bytes,
        "min_capacity": budget.min_capacity,
        "max_capacity": budget.max_capacity,
    }


def _deterministic_available_bytes() -> int:
    """Capacity-budget input: min(total RAM - OS reserve, available now).

    Budgeting from instantaneous available RAM made auto capacity a
    lottery: 56/74/90 slots per layer observed for the same package across
    boots hours apart (page-cache state), and capacity sets the shippable
    hit rate ~linearly. When the system is idle `total - reserve` is the
    binding term, so capacity becomes deterministic (run-to-run
    reproducible); under memory pressure the live available clamps it (never
    budget memory someone else is using). MOESPRESSO_SSD_OS_RESERVE_GB tunes
    the reserve (default 5 GiB for macOS + page cache + apps headroom)."""
    import os

    import psutil

    reserve_gb = float(os.environ.get("MOESPRESSO_SSD_OS_RESERVE_GB", "5"))
    vm = psutil.virtual_memory()
    deterministic = int(vm.total - reserve_gb * (1 << 30))
    # Lower-memory-budget simulation: MOESPRESSO_SSD_MAX_MEMORY_GB (and
    # the --max-memory-gb CLI flags) cap the startup capacity-planner input.
    # This selects expert-pool geometry; it is not an RSS limit. Hit-vs-
    # coverage behavior is representative of that pool capacity, while miss
    # costs remain optimistic when the larger host's page cache satisfies
    # nominally "SSD" reads.
    cap_gb = os.environ.get("MOESPRESSO_SSD_MAX_MEMORY_GB")
    if cap_gb:
        return int(min(deterministic, float(cap_gb) * (1 << 30),
                       vm.available))
    
    # macOS counts reclaimable page cache inconsistently in `available`
    # (observed: an idle system, available wobbling 9.5-11 GB after heavy
    # file IO, still capacity 95 vs 101 across minutes). When live
    # availability is within 25% of the deterministic budget, the gap is
    # reclaimable cache: trust the deterministic number. Only genuine memory
    # pressure (apps holding the RAM) clamps to live availability.
    if vm.available >= deterministic * 0.75:
        return deterministic
    return int(vm.available)


def build_ssd_streaming_model(
    package_dir: str | Path,
    *,
    capacity_per_layer: int | None = None,
    capacity_overrides: Mapping[int, int] | None = None,
    min_resident_experts: int | None = None,
    eviction_policy: str = "lfu",
    seed: int = 42,
):
    """Build `(model, tokenizer, installed)` with routed experts streamed."""
    from mlx_lm.utils import load_config, load_model, load_tokenizer
    from moespresso.runtime.streaming_run_lock import acquire_ssd_streaming_process_lock

    package_dir = Path(package_dir)
    run_lock = acquire_ssd_streaming_process_lock()
    try:
        _silence_false_mistral_warning()
        cfg = load_config(package_dir)
        seed = _mxtq_seed(package_dir, seed)
        manifest = _read_manifest(package_dir)
        from moespresso.runtime.kquant_install import (
            kquant_weight_codec_map_from_manifest,
        )

        has_kquant_dense = bool(
            manifest is not None
            and _kquant_dense_install_enabled()
            and kquant_weight_codec_map_from_manifest(manifest))
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

            budget = package_capacity_budget(
                index=index,
                package_dir=package_dir,
                max_router_fanout=int(text_config.get("num_experts_per_tok", 1)),
                available_bytes=_deterministic_available_bytes(),
                runtime_resident_bytes=router_bf16_f32_resident_bytes(cfg),
            )
            capacity_per_layer = choose_capacity(budget)
            budget_payload = _budget_payload(budget)
        import os as _os_la

        lookahead_env = int(
            _os_la.environ.get("MOESPRESSO_SSD_LOOKAHEAD", "0") or "0")
        spare_slots = 0
        if lookahead_env > 0:
            # The prediction export exists only in the native-gate decode
            # path. Carving demand capacity into spares on a ring/legacy
            # decode path pays the cost for zero benefit.
            from moespresso.runtime.pooled_switchglu import (
                _RING_DECODE,
                _gate_module,
                _ring_visibility_ok,
            )
            gate_live = (_RING_DECODE and _ring_visibility_ok()
                         and _gate_module() is not None)
            if not gate_live:
                print("[ssd-streaming] lookahead requested but the native-"
                      "gate decode path is not live - DISABLED (no capacity "
                      "carved)", flush=True)
                lookahead_env = 0
        if lookahead_env > 0:
            # Spares come out of the capacity budget (same total memory
            # as the baseline): 16 speculative slots vs 16 LFU slots is the
            # honest A/B, and lower demand capacity is exactly the regime
            # where prediction has headroom.
            spare_slots = min(16, max(0, capacity_per_layer - 24))
            capacity_per_layer = capacity_per_layer - spare_slots
        validate_min_resident_experts(
            capacity=int(capacity_per_layer),
            requested=min_resident_experts,
            package_experts=index.num_experts,
        )
        installed = install_pooled_switchglus(
            model,
            package_dir=package_dir,
            index=index,
            capacity_per_layer=capacity_per_layer,
            capacity_overrides=capacity_overrides,
            eviction_policy=eviction_policy,
            seed=seed,
            spare_slots=spare_slots,
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
        # One lookahead decision: `lookahead_env` was gated above (env
        # request, native-gate path live, and spares carved). Re-reading
        # the env here would wire predictors onto a runtime whose log just
        # reported the path disabled: no spares, no prediction export, but
        # worker-side ring polls and a mismatched truth line.
        if lookahead_env > 0:
            wired = install_lookahead(model, lookahead_env)
            object.__setattr__(model, "_moespresso_ssd_lookahead",
                               {"delta": lookahead_env, "wired": wired,
                                "spare_slots": spare_slots})
        # Served prefill chunk for long prompts, the same knob the resident
        # build installs. The default raises the chunk to 4096 for prompts
        # longer than the chunk and keeps the mlx_lm chunk for short prompts;
        # MOESPRESSO_QWEN_PREFILL_CHUNK overrides the value. On the streaming
        # path a larger chunk routes more token-expert pairs per forward pass,
        # so the chunk is priced against the pool residency budget in the
        # campaign measurements.
        from moespresso.runtime.qwen.prefill_chunk import install_prefill_chunk

        install_prefill_chunk(model)

        # Flash D=256 prefill attention for the full-attention layers, the same
        # route the resident build installs. The pooled MoE swap leaves
        # `self_attn` on the full-attention layers untouched, and the attention
        # weights are resident by this point, so the streamed build wraps the
        # identical modules the resident build wraps. The family kill switch
        # MOESPRESSO_QWEN_PREFILL_FLASH_D256=0 disables it on both builds through
        # the one module-level flag; the streamed build adds no knob of its own.
        # A no-op when the kill switch is off or the installed mlx_kquant build
        # lacks the kernel, so an affine-only host serves the stock composed path
        # with no wrapper present.
        from moespresso.runtime.qwen.full_attention import (
            install_flash_prefill_attention,
        )

        flash_wrapped = install_flash_prefill_attention(model)
        object.__setattr__(
            model, "_moespresso_ssd_flash_prefill_layers", int(flash_wrapped))

        # Install the same guarded recurrent-layer fusion as the resident Qwen
        # build. All non-routed weights are hydrated before this wrapper is
        # attached, so it reuses the package-owned modules without changing the
        # loading contract.
        from moespresso.runtime.qwen.gdn_decode import install_fused_gdn_decode

        gdn_wrapped = install_fused_gdn_decode(model)
        object.__setattr__(
            model, "_moespresso_ssd_gdn_fused_layers", int(gdn_wrapped))
        model.eval()
        tokenizer = load_tokenizer(package_dir)
        object.__setattr__(model, "_moespresso_ssd_streaming_lock", run_lock)
        object.__setattr__(model, "_moespresso_ssd_streaming_capacity",
                           int(capacity_per_layer))
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
    lookahead_exports = lookahead_prefetch_loads = 0
    lookahead_ring_misses = lookahead_errors = lookahead_dropped = 0
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
    dontneed = dontneed_errors = 0
    compiled_island_calls = 0
    block_exit_kick_calls = 0
    pipelined_layers = 0
    pipeline_read_seconds = pipeline_join_seconds = 0.0
    bundle_row_preads = bundle_cached_takes = 0
    routed_matmul_calls = routed_matmul_slot_elements = 0
    q6_down_qmv_calls = 0
    routed_projection_matmul_calls = {
        projection: 0 for projection in _SWITCH_PROJECTIONS
    }
    routed_projection_matmul_slot_elements = {
        projection: 0 for projection in _SWITCH_PROJECTIONS
    }
    for layer in _layers(model):
        hc_fused_pre_calls += int(getattr(
            layer, "_moespresso_dsv4_hc_fused_pre_calls", 0) or 0)
        hc_fused_post_calls += int(getattr(
            layer, "_moespresso_dsv4_hc_fused_post_calls", 0) or 0)
        hc_fused_pre_decode_calls += int(getattr(
            layer, "_moespresso_dsv4_hc_fused_pre_decode_calls", 0) or 0)
        hc_fused_pre_tail_decode_calls += int(getattr(
            layer, "_moespresso_dsv4_hc_fused_pre_tail_decode_calls", 0) or 0)
        hc_fused_post_decode_calls += int(getattr(
            layer, "_moespresso_dsv4_hc_fused_post_decode_calls", 0) or 0)
        switch = getattr(getattr(layer, "mlp", None), "switch_mlp", None)
        if not isinstance(switch, PooledSwitchGLU):
            continue
        modules += 1
        # the layer's three pools share one BundleRowCache; count it once
        row_cache = switch.gate_proj.pool.row_cache
        if row_cache is not None:
            bundle_row_preads += row_cache.total_preads
            bundle_cached_takes += row_cache.total_cached_takes
        calls += switch.total_calls
        decode_calls += switch.decode_calls
        prefill_calls += switch.prefill_calls
        direct_calls += switch.direct_calls
        row_chunked_calls += switch.row_chunked_calls
        sorted_chunked_calls += switch.sorted_chunked_calls
        segmented_prefill_calls += switch.segmented_prefill_calls
        unified_sorted_prefill_calls += getattr(
            switch, "unified_sorted_prefill_calls", 0)
        barrier_free_prefill_calls += getattr(
            switch, "barrier_free_prefill_calls", 0)
        barrier_free_identity_calls += getattr(
            switch, "barrier_free_identity_calls", 0)
        barrier_free_fused_swiglu_calls += getattr(
            switch, "barrier_free_fused_swiglu_calls", 0)
        barrier_free_decode_calls += getattr(
            switch, "barrier_free_decode_calls", 0)
        barrier_free_decode_flush_calls += getattr(
            switch, "barrier_free_decode_flush_calls", 0)
        decode_routed_fused_calls += getattr(
            switch, "decode_routed_fused_calls", 0)
        pipelined_decode_fused_calls += getattr(
            switch, "pipelined_decode_fused_calls", 0)
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
        overlap_skipped_over_capacity_calls += (
            switch.overlap_skipped_over_capacity_calls)
        overlap_ticket_mismatch_calls += switch.overlap_ticket_mismatch_calls
        prefetch_ticket_submitted += switch.prefetch_ticket_submitted
        prefetch_ticket_consumed += switch.prefetch_ticket_consumed
        prefetch_ticket_mismatched += switch.prefetch_ticket_mismatched
        prefetch_ticket_stale += switch.prefetch_ticket_stale
        prefetch_ticket_experts += switch.prefetch_ticket_experts
        prefetch_ticket_loaded += switch.prefetch_ticket_loaded
        prefetch_ticket_wait_seconds += switch.prefetch_ticket_wait_seconds
        lookahead_exports += getattr(switch, "lookahead_exports", 0)
        lookahead_prefetch_loads += getattr(
            switch, "lookahead_prefetch_loads", 0)
        lookahead_ring_misses += getattr(switch, "lookahead_ring_misses", 0)
        lookahead_errors += getattr(switch, "lookahead_errors", 0)
        lookahead_dropped += getattr(switch, "lookahead_dropped", 0)
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
        routed_weighted_sum_slot_elements += (
            switch.routed_weighted_sum_slot_elements)
        routed_weighted_sum_output_elements += (
            switch.routed_weighted_sum_output_elements)
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
            dontneed += pool.total_dontneed
            dontneed_errors += pool.total_dontneed_errors
            expert_spec_prefetch_loads += pool.total_prefetch_loads
            expert_spec_prefetch_skips += pool.total_prefetch_skips
        for projection, module in _projection_modules_for_switch(switch):
            matmul_calls = int(getattr(module, "matmul_slot_calls", 0))
            matmul_elements = int(getattr(module, "matmul_slot_elements", 0))
            routed_matmul_calls += matmul_calls
            routed_matmul_slot_elements += matmul_elements
            routed_projection_matmul_calls[projection] += matmul_calls
            routed_projection_matmul_slot_elements[projection] += matmul_elements
            q6_down_qmv_calls += int(
                getattr(module, "decode_q6_qmv_calls", 0)
            )

    # The DS4 ratio-4 prefill consumer choice happens inside the kernel
    # module, so its engagement counts are module totals rather than
    # per-layer attributes; the speed-stats runner consumes them as
    # before/after deltas. The tiled score operand form and the grouped
    # wo_a projection form export the same way.
    from moespresso.runtime.deepseek_v4.indexed_attention_kernel import (
        indexer_scores_call_counts,
        prefill_consumer_call_counts,
    )
    from moespresso.runtime.deepseek_v4.model import (
        affine_wo_fp32_call_counts,
        attention_seam_rope_call_counts,
        banded_prefill_call_counts,
        q8_dense_matmul_call_counts,
        router_gate_trim_call_counts,
        wo_a_projection_call_counts,
    )

    consumer_counts = prefill_consumer_call_counts()
    scores_counts = indexer_scores_call_counts()
    wo_a_counts = wo_a_projection_call_counts()
    banded_counts = banded_prefill_call_counts()
    seam_rope_counts = attention_seam_rope_call_counts()
    router_trim_counts = router_gate_trim_call_counts()
    q8_dense_counts = q8_dense_matmul_call_counts()
    affine_wo_counts = affine_wo_fp32_call_counts()

    # Flash D=256 prefill engagement, the same route the resident build installs;
    # the streamed build wraps the identical `self_attn` modules. Reachable on
    # the streamed model whether or not the wrapper is present (empty when the
    # kill switch is off).
    from moespresso.runtime.qwen.full_attention import (
        flash_prefill_attention_stats,
    )

    flash_counts = flash_prefill_attention_stats(model)

    from moespresso.runtime.qwen.gdn_decode import fused_gdn_decode_stats

    gdn_counts = fused_gdn_decode_stats(model)

    from moespresso.runtime.qwen.router_gemv import router_bf16_f32_stats

    router_gemv_counts = router_bf16_f32_stats(model)

    total = hits + misses
    return {
        "enabled": modules > 0,
        "kquant_dense_modules_installed": int(getattr(
            model, "_moespresso_ssd_kquant_dense_installed", 0) or 0),
        "flash_prefill_wrapped_layers": flash_counts["wrapped_layers"],
        "flash_prefill_calls": flash_counts["flash_calls"],
        "q8_decode_tile16_calls": flash_counts["decode_calls"],
        "q8_decode_dimension_merge_calls": flash_counts[
            "decode_dimension_merge_calls"
        ],
        "flash_prefill_fallback_disabled": flash_counts[
            "fallback_prefill_disabled"
        ],
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
        "gdn_fused_fallback_disabled": gdn_counts["fallback_disabled"],
        "gdn_fused_fallback_training": gdn_counts["fallback_training"],
        "gdn_fused_fallback_input": gdn_counts["fallback_input"],
        "gdn_fused_fallback_mask": gdn_counts["fallback_mask"],
        "gdn_fused_fallback_cache": gdn_counts["fallback_cache"],
        "gdn_fused_fallback_geometry": gdn_counts["fallback_geometry"],
        "gdn_fused_fallback_dtype": gdn_counts["fallback_dtype"],
        "gdn_fused_fallback_kernel": gdn_counts["fallback_kernel"],
        "gdn_fused_fallback_qkv": gdn_counts["fallback_qkv"],
        "router_bf16_f32_wrapped_layers": router_gemv_counts["wrapped_layers"],
        "router_bf16_f32_validated_layers": router_gemv_counts[
            "validated_layers"
        ],
        "router_bf16_f32_kernel_calls": router_gemv_counts["kernel_calls"],
        "router_bf16_f32_fallback_disabled": router_gemv_counts[
            "fallback_disabled"
        ],
        "router_bf16_f32_fallback_training": router_gemv_counts[
            "fallback_training"
        ],
        "router_bf16_f32_fallback_input_shape": router_gemv_counts[
            "fallback_input_shape"
        ],
        "router_bf16_f32_fallback_input_dtype": router_gemv_counts[
            "fallback_input_dtype"
        ],
        "router_bf16_f32_fallback_weight_contract": router_gemv_counts[
            "fallback_weight_contract"
        ],
        "router_bf16_f32_fallback_kernel": router_gemv_counts["fallback_kernel"],
        "r4_prefill_consumer_mma_calls": consumer_counts["mma"],
        "r4_prefill_consumer_scalar_calls": (
            consumer_counts["v2"] + consumer_counts["v1"]),
        "r4_prefill_scores_f16_calls": scores_counts["f16"],
        "r4_prefill_scores_f32_calls": scores_counts["f32"],
        "wo_a_batched_decode_calls": wo_a_counts["batched_decode"],
        "wo_a_gather_decode_calls": wo_a_counts["gather_decode"],
        "wo_a_loop_projection_calls": wo_a_counts["loop"],
        "q8_dense_decode_qmv_calls": q8_dense_counts["decode_qmv"],
        "q8_dense_decode_wire_qmv_wo_b_calls": (
            q8_dense_counts["decode_wire_qmv_wo_b"]),
        "q8_dense_decode_wire_qmv_lm_head_calls": (
            q8_dense_counts["decode_wire_qmv_lm_head"]),
        "q8_dense_prefill_dequant_calls": q8_dense_counts["prefill_dequant"],
        "affine_wo_fp32_wo_a_calls": affine_wo_counts["wo_a"],
        "affine_wo_fp32_wo_b_calls": affine_wo_counts["wo_b"],
        "affine_wo_fp32_delegated_calls": affine_wo_counts["delegated"],
        "banded_prefill_mma_calls": banded_counts["mma"],
        "banded_prefill_sdpa_calls": banded_counts["sdpa"],
        "banded_prefill_mma_offset_calls": banded_counts["mma_offset"],
        "banded_prefill_composed_offset_calls": banded_counts["composed_offset"],
        "attn_seam_rope_fused_calls": seam_rope_counts["fused"],
        "attn_seam_rope_composed_calls": seam_rope_counts["composed"],
        "router_gate_precast_calls": router_trim_counts["precast"],
        "router_gate_select_kernel_calls": router_trim_counts["select_kernel"],
        "router_gate_select_composed_calls": (
            router_trim_counts["select_composed"]),
        "router_gate_composed_calls": router_trim_counts["composed"],
        "switch_modules": modules,
        "resident_slots": resident_slots,
        "expert_hits": hits,
        "expert_misses": misses,
        "expert_loads": loads,
        "expert_evictions": evictions,
        "expert_load_seconds": load_seconds,
        # IO shape: a miss costs one bundle-row pread (shared across the
        # layer's three pools); cached_takes are the dedup hits. preads is the
        # number the format lane's A/B measures (was ~6x loads before).
        "bundle_row_preads": bundle_row_preads,
        "bundle_cached_takes": bundle_cached_takes,
        # Page-cache hygiene: advisory MADV_DONTNEED on evicted rows
        "expert_dontneed": dontneed,
        "expert_dontneed_errors": dontneed_errors,
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
        "lookahead_exports": lookahead_exports,
        "lookahead_prefetch_loads": lookahead_prefetch_loads,
        "lookahead_ring_misses": lookahead_ring_misses,
        "lookahead_errors": lookahead_errors,
        "lookahead_dropped": lookahead_dropped,
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
        "routed_gate_matmul_slot_elements": (
            routed_projection_matmul_slot_elements["gate_proj"]),
        "routed_up_matmul_slot_elements": (
            routed_projection_matmul_slot_elements["up_proj"]),
        "routed_down_matmul_slot_elements": (
            routed_projection_matmul_slot_elements["down_proj"]),
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
        "capacity_per_layer": getattr(
            model, "_moespresso_ssd_streaming_capacity", None),
        "capacity_overrides": getattr(
            model, "_moespresso_ssd_streaming_capacity_overrides", {}),
        "eviction_policy": getattr(
            model, "_moespresso_ssd_streaming_eviction_policy", None),
        "capacity_budget": getattr(
            model, "_moespresso_ssd_streaming_capacity_budget", None),
        "adaptive_growth": getattr(
            model, "_moespresso_ssd_streaming_adaptive_growth", None),
    }


def suggest_capacity_overrides_from_layer_stats(
    rows: list[dict],
    *,
    extra_slot_budget: int | None = None,
    extra_byte_budget: int | None = None,
    target: str = "all",
) -> dict[int, int]:
    """Greedily spend extra residency budget where observed churn is highest."""
    if (extra_slot_budget is None) == (extra_byte_budget is None):
        raise ValueError(
            "pass exactly one of extra_slot_budget or extra_byte_budget")
    if target not in {"all", "decode"}:
        raise ValueError("target must be 'all' or 'decode'")

    remaining_slots = (
        int(extra_slot_budget)
        if extra_slot_budget is not None
        else None
    )
    remaining_bytes = (
        int(extra_byte_budget)
        if extra_byte_budget is not None
        else None
    )
    if remaining_slots is not None and remaining_slots <= 0:
        return {}
    if remaining_bytes is not None and remaining_bytes <= 0:
        return {}

    def _row_cost(row: dict) -> int:
        cost = int(row.get("slot_bytes", 0))
        if cost <= 0:
            raise ValueError(
                "extra_byte_budget requires positive slot_bytes in every row")
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
        need = target_capacity - current
        if need <= 0:
            continue
        candidates.append((
            int(row.get("expert_loads", 0)),
            int(row.get("expert_misses", 0)),
            target_capacity,
            int(row["layer"]),
            current,
            need,
            _row_cost(row) if remaining_bytes is not None else 1,
        ))

    overrides: dict[int, int] = {}
    for _loads, _misses, _target, layer, current, need, slot_bytes in sorted(
        candidates,
        reverse=True,
    ):
        if remaining_slots is not None:
            if remaining_slots <= 0:
                break
            grant = min(need, remaining_slots)
            remaining_slots -= grant
        elif remaining_bytes is not None:
            if remaining_bytes < slot_bytes:
                continue
            grant = min(need, remaining_bytes // slot_bytes)
            remaining_bytes -= grant * slot_bytes
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
    for layer_idx, layer in enumerate(_layers(model)):
        switch = getattr(getattr(layer, "mlp", None), "switch_mlp", None)
        if not isinstance(switch, PooledSwitchGLU):
            continue
        if layer_idx not in requested:
            continue
        capacity = requested[layer_idx]
        current = min(
            pool.capacity for pool in _unique_projection_pools_for_switch(switch)
        )
        if capacity <= current:
            continue
        switch.grow_capacity(capacity)
        if seed_hot:
            switch.seed_hot_free_slots()
        applied[layer_idx] = capacity

    if applied:
        current_overrides = dict(getattr(
            model,
            "_moespresso_ssd_streaming_capacity_overrides",
            {},
        ))
        current_overrides.update(applied)
        object.__setattr__(
            model,
            "_moespresso_ssd_streaming_capacity_overrides",
            current_overrides,
        )
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

    The live memory floor (min_available_bytes) is the safety
    contract; this cap is a secondary bound on how much the pools may ever
    grow beyond their build-time capacity. The old 512 MiB default bound
    first on the residency-starved shippable package (hit rate tracks
    capacity ~linearly), leaving free-above-floor RAM
    unused. Default raised to 2 GiB: growth still only takes what the
    floor allows, one conservative step after each request.
    MOESPRESSO_SSD_GROWTH_MAX_EXTRA_GB overrides (0 disables growth)."""
    import os

    raw = os.environ.get("MOESPRESSO_SSD_GROWTH_MAX_EXTRA_GB")
    if raw is not None:
        return int(float(raw) * (1 << 30))
    return 2 << 30


def maybe_adapt_ssd_streaming_capacity(
    model,
    *,
    available_bytes: int | None = None,
    min_available_bytes: int = 4 << 30,
    max_extra_bytes: int | None = None,
    seed_hot: bool = True,
) -> dict:
    """Conservatively grow hot routed layers after real request evidence exists."""
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
        extra_byte_budget=extra_byte_budget,
        target="all",
    )
    applied = grow_ssd_streaming_capacity(model, plan, seed_hot=seed_hot)
    after_rows = ssd_streaming_layer_stats(model)
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


def default_saved_hotlist_path(package_dir: str | Path) -> Path:
    """Canonical per-artifact location for the runtime-saved demand hotlist.

    Keyed by the package manifest's content-addressed id (so a re-converted
    or different package never inherits another's demand); falls back to the
    package dir name when no manifest is readable. Lives under the user
    cache. The package directory may be a read-only mount and is never used for
    generated cache data;
    MOESPRESSO_HOTLIST_DIR overrides the directory (tests, multi-user)."""
    import os

    package_dir = Path(package_dir)
    key = package_dir.name
    manifest_path = package_dir / "package_manifest.json"
    try:
        artifact_id = json.loads(manifest_path.read_text()).get("artifact_id", "")
        if artifact_id.startswith("pkg:"):
            key = artifact_id[len("pkg:"):][:24]
    except (OSError, ValueError):
        pass
    base = os.environ.get("MOESPRESSO_HOTLIST_DIR")
    root = Path(base) if base else Path.home() / ".cache" / "moespresso" / "hotlists"
    return root / f"{key}.json"


def _all_pools_at_full_capacity(model) -> bool:
    """True when every routed projection pool can hold its full expert set."""
    pools_seen = False
    for layer in _layers(model):
        switch = getattr(getattr(layer, "mlp", None), "switch_mlp", None)
        if not isinstance(switch, PooledSwitchGLU):
            continue
        for pool in _unique_projection_pools_for_switch(switch):
            pools_seen = True
            if pool.capacity < pool.num_experts:
                return False
    return pools_seen


def seed_expert_residency(model, package_dir: str | Path) -> dict:
    """Layered cold-start seeding.

    Precedence: an explicit MOESPRESSO_SSD_PREWARM_EXPERTS=all request wins,
    and MOESPRESSO_SSD_PREWARM_EXPERTS=none explicitly skips both the prewarm
    and the full-capacity default (bounded-budget gate runs use it); then,
    when every projection pool's capacity covers the full expert set,
    the default prewarms all experts at load (source `all-default`, kill
    switch MOESPRESSO_SSD_PREWARM_DEFAULT=0). Pool residency selects the
    routed prefill kernel, so a cold pool at full capacity would serve the
    segmented numerics until the pools fill and diverge from the
    gate-certified barrier-free path at knife-edge tokens; prewarming at load
    pins serving to the gate-certified numerics and removes the cold
    first-request SSD reads. Measured on the byte-faithful DS4 package, the
    prewarm costs ~14 s of load time (2.9 s lazy vs 16.9 s prewarm-all) and
    loads faster than saved-hotlist seeding of the same expert set (24.5 s).

    Below full capacity the hotlist tiers apply, kill switch
    MOESPRESSO_SSD_HOTLIST=0. Source precedence, measured on real demand
    (top-70 demand mass): runtime-saved demand from a prior session (~0.60) >
    the package's imatrix-counts hotlist (~0.40) > nothing (~0.27 arbitrary).
    Hotlist seeding only fills free slots and the installed prior is capped
    (load_expert_hotlist), so it can never evict live demand nor outvote it
    for long."""
    import os

    from moespresso.package.hotlist import HOTLIST_NAME

    info = {"source": None, "path": None, "seeded": 0,
            "save_path": str(default_saved_hotlist_path(package_dir))}
    prewarm = os.environ.get("MOESPRESSO_SSD_PREWARM_EXPERTS", "").strip().lower()
    if prewarm and prewarm != "none":
        if prewarm != "all":
            raise SSDStreamingBuildError(
                "MOESPRESSO_SSD_PREWARM_EXPERTS must be 'all' or 'none' when set")
        info["source"] = "all"
        info["seeded"] = seed_all_expert_residency(model)
        return info
    # 'none' is the explicit no-prewarm override: it skips the full-capacity
    # default below and falls through to the hotlist tiers, so callers that
    # pin the prewarm by default (the quality gates) can still run a bounded
    # capacity budget when asked.
    if (prewarm != "none"
            and os.environ.get("MOESPRESSO_SSD_PREWARM_DEFAULT", "1") != "0"
            and _all_pools_at_full_capacity(model)):
        info["source"] = "all-default"
        info["seeded"] = seed_all_expert_residency(model)
        return info
    if os.environ.get("MOESPRESSO_SSD_HOTLIST", "1") == "0":
        info["source"] = "disabled"
        return info
    package_dir = Path(package_dir)
    candidates = (
        ("saved", default_saved_hotlist_path(package_dir)),
        ("package", package_dir / HOTLIST_NAME),
    )
    for source, path in candidates:
        if path.exists():
            info["seeded"] = load_expert_hotlist(model, path)
            info["source"] = source
            info["path"] = str(path)
            break
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
        switch = getattr(getattr(layer, "mlp", None), "switch_mlp", None)
        if not isinstance(switch, PooledSwitchGLU):
            continue
        pools = list(_unique_projection_pools_for_switch(switch))
        num_experts = pools[0].num_experts
        for pool in pools:
            if pool.num_experts != num_experts:
                raise SSDStreamingBuildError(
                    f"layer {layer_idx}: projection pools disagree on expert count")
            if pool.capacity < num_experts:
                raise SSDStreamingBuildError(
                    f"layer {layer_idx}: full expert prewarm requires capacity "
                    f"{num_experts}, got {pool.capacity}")

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


def save_expert_hotlist(model, path: str | Path) -> int:
    """Persist per-layer expert demand (the gate pool's touch frequencies) so
    a later session can warm-start residency.

    Returns the number of layers written."""
    hotlist: dict[str, dict[str, int]] = {}
    for layer_idx, layer in enumerate(_layers(model)):
        switch = getattr(getattr(layer, "mlp", None), "switch_mlp", None)
        if not isinstance(switch, PooledSwitchGLU):
            continue
        freq = switch.gate_proj.pool._freq
        if freq:
            hotlist[str(layer_idx)] = {
                str(expert): int(count) for expert, count in freq.items()
            }
    payload = {"version": 1, "kind": "expert_hotlist", "layers": hotlist}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return len(hotlist)


def load_expert_hotlist(model, path: str | Path, *, seed: bool = True,
                        prior_cap: int | None = 8) -> int:
    """Warm-start residency from a hotlist: install the demand counts into
    all three pools of each layer and (by default) seed the hottest experts
    into free slots now, moving cold-start misses into build time.

    Works for both hotlist sources (same schema): runtime-saved demand
    (save_expert_hotlist) and the package's imatrix-derived cold-start list
    (package/hotlist.py). Seeding ranks on the raw counts (precise order);
    afterwards the installed prior is rescaled so no entry exceeds
    `prior_cap`: imatrix counts run to the hundreds of thousands and a raw
    install would make seeded experts effectively un-evictable, poisoning
    LFU adaptation (live traffic must be able to overtake a stale prior
    within a few touches). prior_cap=None disables the rescale.

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
        switch = getattr(getattr(layer, "mlp", None), "switch_mlp", None)
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
                    pool._freq = {
                        e: max(1, round(c * scale))
                        for e, c in pool._freq.items()
                    }
    return seeded


def ssd_streaming_layer_stats(model) -> list[dict]:
    """Per-routed-layer residency and miss counters for speed diagnosis."""
    rows = []
    for layer_idx, layer in enumerate(_layers(model)):
        switch = getattr(getattr(layer, "mlp", None), "switch_mlp", None)
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
        routed_projection_matmul_calls = {
            projection: 0 for projection in _SWITCH_PROJECTIONS
        }
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
            q6_down_qmv_calls += int(
                getattr(module, "decode_q6_qmv_calls", 0)
            )

        total = hits + misses
        capacity = min(
            pool.capacity for pool in _unique_projection_pools_for_switch(switch)
        )
        projection_pool_count = len(_unique_projection_pools_for_switch(switch))
        num_experts = switch.gate_proj.pool.num_experts
        rows.append({
            "layer": layer_idx,
            "capacity": capacity,
            "num_experts": num_experts,
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
            "unified_sorted_prefill_calls": getattr(
                switch, "unified_sorted_prefill_calls", 0),
            "barrier_free_prefill_calls": getattr(
                switch, "barrier_free_prefill_calls", 0),
            "barrier_free_identity_calls": getattr(
                switch, "barrier_free_identity_calls", 0),
            "barrier_free_fused_swiglu_calls": getattr(
                switch, "barrier_free_fused_swiglu_calls", 0),
            "barrier_free_decode_calls": getattr(
                switch, "barrier_free_decode_calls", 0),
            "barrier_free_decode_flush_calls": getattr(
                switch, "barrier_free_decode_flush_calls", 0),
            "decode_routed_fused_calls": getattr(
                switch, "decode_routed_fused_calls", 0),
            "pipelined_decode_fused_calls": getattr(
                switch, "pipelined_decode_fused_calls", 0),
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
            "overlap_skipped_over_capacity_calls": (
                switch.overlap_skipped_over_capacity_calls),
            "overlap_ticket_mismatch_calls": switch.overlap_ticket_mismatch_calls,
            "prefetch_ticket_submitted": switch.prefetch_ticket_submitted,
            "prefetch_ticket_consumed": switch.prefetch_ticket_consumed,
            "prefetch_ticket_mismatched": switch.prefetch_ticket_mismatched,
            "prefetch_ticket_stale": switch.prefetch_ticket_stale,
            "prefetch_ticket_experts": switch.prefetch_ticket_experts,
            "prefetch_ticket_loaded": switch.prefetch_ticket_loaded,
            "prefetch_ticket_wait_seconds": switch.prefetch_ticket_wait_seconds,
            "lookahead_exports": getattr(switch, "lookahead_exports", 0),
            "lookahead_prefetch_loads": getattr(
                switch, "lookahead_prefetch_loads", 0),
            "lookahead_ring_misses": getattr(
                switch, "lookahead_ring_misses", 0),
            "lookahead_errors": getattr(switch, "lookahead_errors", 0),
            "lookahead_dropped": getattr(switch, "lookahead_dropped", 0),
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
            "routed_weighted_sum_slot_elements": (
                switch.routed_weighted_sum_slot_elements),
            "routed_weighted_sum_output_elements": (
                switch.routed_weighted_sum_output_elements),
            "routed_gate_matmul_calls": routed_projection_matmul_calls["gate_proj"],
            "routed_up_matmul_calls": routed_projection_matmul_calls["up_proj"],
            "routed_down_matmul_calls": routed_projection_matmul_calls["down_proj"],
            "q6_down_qmv_calls": q6_down_qmv_calls,
            "routed_gate_matmul_slot_elements": (
                routed_projection_matmul_slot_elements["gate_proj"]),
            "routed_up_matmul_slot_elements": (
                routed_projection_matmul_slot_elements["up_proj"]),
            "routed_down_matmul_slot_elements": (
                routed_projection_matmul_slot_elements["down_proj"]),
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
        })
    return rows
