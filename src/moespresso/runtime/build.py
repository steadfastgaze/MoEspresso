"""Build model and tokenizer instances from declared package formats.

Family adapters bind the model graph to packed tensors and routed-expert pools.
Dense affine modules use the package-generated Jang configuration and tensor map.
Runtime dependencies are imported lazily.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

# The pinned Transformers releases emit this warning for non-Mistral tokenizers
# loaded beside model files. The tokenizer loads as ``Qwen2Tokenizer`` and its
# tokenization is unaffected, so filter that message without changing
# ``fix_mistral_regex``.
class _DropMistralRegexWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "incorrect regex" not in msg and "fix_mistral_regex" not in msg


class _DropDeepSeekV4RopeWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage() != (
            "Unrecognized keys in `rope_parameters` for 'rope_type'='default': "
            "{'attention_factor'}"
        )


class _DropQwen4ConfigTypeWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage() not in {
            "You are using a model of type `qwen4_exp` to instantiate a model "
            "of type ``. This is not supported for all configurations of models "
            "and can yield errors.",
            "You are using a model of type `qwen4_exp` to instantiate a model "
            "of type ``. This may be expected if you are loading a checkpoint that "
            "shares a subset of the architecture (e.g., loading a `sam2_video` "
            "checkpoint into `Sam2Model`), but is otherwise not supported and can "
            "yield errors. Please verify that the checkpoint is compatible with the "
            "model you are instantiating.",
        }


def _silence_known_transformers_warnings() -> None:
    # The message is emitted by logger.warning in transformers'
    # tokenization_utils_tokenizers module, so the filter must sit on that logger:
    # a filter on the parent 'transformers' logger does not catch child records.
    lg = logging.getLogger("transformers.tokenization_utils_tokenizers")
    if not any(isinstance(f, _DropMistralRegexWarning) for f in lg.filters):
        lg.addFilter(_DropMistralRegexWarning())

    # AutoTokenizer loads the model config while selecting a tokenizer. The
    # Transformers DeepSeek-V4 config adds attention_factor itself, then warns
    # about that field during its own RoPE validation.
    lg = logging.getLogger("transformers.modeling_rope_utils")
    if not any(isinstance(f, _DropDeepSeekV4RopeWarning) for f in lg.filters):
        lg.addFilter(_DropDeepSeekV4RopeWarning())

    lg = logging.getLogger("transformers.configuration_utils")
    if not any(isinstance(f, _DropQwen4ConfigTypeWarning) for f in lg.filters):
        lg.addFilter(_DropQwen4ConfigTypeWarning())


def _qwen_eos_token_ids(config: dict) -> set[int] | None:
    """Return every declared Qwen stop id from the merged model config."""
    stop_ids: set[int] = set()

    def collect(value) -> None:
        if value is None:
            return
        if isinstance(value, int):
            stop_ids.add(value)
            return
        stop_ids.update(int(token_id) for token_id in value)

    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        collect(text_config.get("eos_token_id"))
    collect(config.get("eos_token_id"))
    return stop_ids or None


def _apply_tensor_map(model, tensor_map: dict) -> None:
    """Override bits/group_size on QuantizedLinear modules from the explicit map.

    Jang builds modules from config.json's quantization block. The explicit map
    sets each module's bits and group size so module construction does not infer
    an incorrect precision."""
    for name, module in model.named_modules():
        if name in tensor_map and hasattr(module, "bits"):
            alloc = tensor_map[name]
            module.bits = alloc["bits"]
            module.group_size = alloc["group_size"]


class MixedBitSwitchGLUError(RuntimeError):
    pass


def _routed_expert_index(package_dir: Path):
    """The package's expert index, validated (None when no shards on disk)."""
    if not list(package_dir.glob("model-*.safetensors")):
        return None

    from moespresso.runtime.expert_index import build_expert_index

    index = build_expert_index(package_dir)
    problems = index.validate()
    if problems:
        raise MixedBitSwitchGLUError(
            "invalid routed-expert metadata: " + "; ".join(problems))
    return index


def _mixed_gate_up_layers_from_headers(package_dir: Path) -> set[int]:
    """Return layers whose routed gate/up bit widths differ, using headers only."""
    index = _routed_expert_index(package_dir)
    return set() if index is None else _mixed_gate_up_layers(index)


def _mixed_gate_up_layers(index) -> set[int]:
    mixed: set[int] = set()
    for layer in index.layers_indexed():
        if not index.has_projection(layer=layer, projection="gate_proj"):
            continue
        gate_bits = index.bits(layer=layer, projection="gate_proj")
        up_bits = index.bits(layer=layer, projection="up_proj")
        if gate_bits != up_bits:
            mixed.add(layer)
    return mixed


class RoutedExpertInstallError(RuntimeError):
    pass


def _decoder_layers(model):
    for path in (
        ("language_model", "model", "layers"),
        ("model", "layers"),
        ("layers",),
    ):
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    return None


def _install_routed_experts_from_bundles(model, package_dir: Path, index) -> int:
    """Install resident K-quant projections from validated bundle geometry.

    Each layer is read once. Component bytes populate the projection buffers
    before replacing the graph's routed modules. Missing data fails loading.
    """
    import mlx.core as mx
    from moespresso.package.bundle import KQUANT_CODEC

    layers = _decoder_layers(model)
    if layers is None:
        raise RoutedExpertInstallError("could not find decoder layers on model")

    installed = 0
    n_exp = index.num_experts
    for layer_idx in index.layers_indexed():
        if layer_idx >= len(layers):
            raise RoutedExpertInstallError(
                f"expert index declares layer {layer_idx}, model has {len(layers)}")
        sw = getattr(getattr(layers[layer_idx], "mlp", None), "switch_mlp", None)
        if sw is None:
            raise RoutedExpertInstallError(
                f"expert index declares layer {layer_idx}, but the loaded model "
                "has no switch_mlp there")
        br0 = index.locate_row(layer=layer_idx, expert=0)
        row_bytes = index.row_bytes(layer=layer_idx)
        with open(package_dir / br0.shard, "rb") as f:
            f.seek(br0.offset)
            raw = f.read(n_exp * row_bytes)
        if len(raw) != n_exp * row_bytes:
            raise RoutedExpertInstallError(
                f"layer {layer_idx}: short bundle read "
                f"({len(raw)} of {n_exp * row_bytes} bytes)")
        rows = memoryview(raw)
        comps = index.row_components(layer=layer_idx)
        for proj in ("gate_proj", "up_proj", "down_proj"):
            geo = index.geometry(layer=layer_idx, projection=proj)
            if geo.codec == KQUANT_CODEC:
                from mlx_kquant.nn import KQuantSwitchLinear

                bytes_per_block = int(geo.bytes_per_block or 0)
                weights_per_block = int(geo.weights_per_block or 0)
                if bytes_per_block <= 0 or weights_per_block <= 0:
                    raise RoutedExpertInstallError(
                        f"layer {layer_idx} {proj}: missing K-quant geometry")
                if geo.packed_cols % bytes_per_block:
                    raise RoutedExpertInstallError(
                        f"layer {layer_idx} {proj}: K-quant bytes_per_row "
                        f"{geo.packed_cols} is not divisible by {bytes_per_block}")
                in_features = geo.packed_cols // bytes_per_block * weights_per_block
                mod = KQuantSwitchLinear(
                    n_exp, geo.out_features, in_features, False, geo.kquant_codec)
                weight_component = "weight"
            else:
                raise RoutedExpertInstallError(
                    f"layer {layer_idx} {proj}: unsupported routed expert "
                    f"codec {geo.codec!r}")
            # Fill persistent MLX buffers by byte copy (the pool pattern: no
            # numpy on the engine path).
            weight_dtype = mx.uint8
            weight = mx.zeros(
                (n_exp, geo.out_features, geo.packed_cols),
                dtype=weight_dtype,
            )
            weight_view = memoryview(weight).cast("B")
            wc = comps[(proj, weight_component)]
            wn = wc["nbytes"]
            for e in range(n_exp):
                row = rows[e * row_bytes:(e + 1) * row_bytes]
                weight_view[e * wn:(e + 1) * wn] = (
                    row[wc["offset"]:wc["offset"] + wn])
            mod.weight = weight
            mod.scales = mx.zeros((1,), dtype=mx.uint8)
            mx.eval(mod.weight, mod.scales)
            setattr(sw, proj, mod)
        del rows, raw
        installed += 1
    return installed


def _load_qwen_kquant_model(
    manifest: dict,
    package_dir: Path,
    *,
    load_config_fn=None,
    load_model_fn=None,
    load_tokenizer_fn=None,
    install_kquant_modules_fn=None,
    load_non_routed_fn=None,
):
    """Build a Qwen MoE skeleton and hydrate manifest-declared K-quant tensors."""
    if load_config_fn is None or load_tokenizer_fn is None:
        from mlx_lm.utils import load_config, load_tokenizer

        load_config_fn = load_config if load_config_fn is None else load_config_fn
        load_tokenizer_fn = (
            load_tokenizer if load_tokenizer_fn is None else load_tokenizer_fn)
    if load_model_fn is None:
        from mlx_lm.utils import _get_classes

        def load_model_fn(package_dir, *, lazy, strict, model_config):
            del package_dir, strict
            model_class, model_args_class = _get_classes(config=model_config)
            model = model_class(model_args_class.from_dict(model_config))
            model.eval()
            if not lazy:
                import mlx.core as mx

                mx.eval(model.parameters())
            return model, model_config
    if install_kquant_modules_fn is None:
        from moespresso.runtime.kquant_install import install_manifest_kquant_modules

        install_kquant_modules_fn = install_manifest_kquant_modules
    if load_non_routed_fn is None:
        from moespresso.runtime.ssd_streaming_build import _load_non_routed_resident

        load_non_routed_fn = _load_non_routed_resident

    model_config = dict(load_config_fn(package_dir))
    # load_model() merges this dict into config.json, so absence is not enough:
    # explicitly disable MLX affine quantization before installing K-quant leaves.
    model_config["quantization"] = None
    model_config["quantization_config"] = None
    model, _model_config = load_model_fn(
        package_dir,
        lazy=True,
        strict=False,
        model_config=model_config,
    )
    install_kquant_modules_fn(model, manifest)
    load_non_routed_fn(model, package_dir)
    tokenizer = load_tokenizer_fn(
        package_dir,
        eos_token_ids=_qwen_eos_token_ids(model_config),
    )
    return model, tokenizer


def _wrap_mixed_bit_switchglus(
    model,
    *,
    required_mixed_layers: set[int] | None = None,
) -> int:
    """Bypass jang's class-level fused SwitchGLU patch for mixed gate/up bits."""
    from moespresso.runtime.owned_switchglu import OwnedSwitchGLU

    required_mixed_layers = required_mixed_layers or set()
    wrapped = 0
    layers = _decoder_layers(model)
    if layers is None:
        layers = []
    seen_required: set[int] = set()
    for layer_idx, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        sw = getattr(mlp, "switch_mlp", None)
        if sw is None:
            if layer_idx in required_mixed_layers:
                raise MixedBitSwitchGLUError(
                    f"metadata declares mixed gate/up bits for layer {layer_idx}, "
                    "but the loaded model has no switch_mlp there")
            continue
        gate = getattr(sw, "gate_proj", None)
        up = getattr(sw, "up_proj", None)
        down = getattr(sw, "down_proj", None)
        if gate is None or up is None or down is None:
            if layer_idx in required_mixed_layers:
                raise MixedBitSwitchGLUError(
                    f"metadata declares mixed gate/up bits for layer {layer_idx}, "
                    "but the loaded switch_mlp is missing gate/up/down projections")
            continue
        gate_bits = getattr(gate, "bits", None)
        up_bits = getattr(up, "bits", None)
        if gate_bits is None or up_bits is None:
            if layer_idx in required_mixed_layers:
                raise MixedBitSwitchGLUError(
                    f"metadata declares mixed gate/up bits for layer {layer_idx}, "
                    "but the loaded projections do not expose .bits")
            continue
        if gate_bits == up_bits:
            if layer_idx in required_mixed_layers:
                raise MixedBitSwitchGLUError(
                    f"metadata declares mixed gate/up bits for layer {layer_idx}, "
                    f"but loaded gate/up bits are both {gate_bits}")
            continue
        setattr(mlp, "switch_mlp", OwnedSwitchGLU(
            gate_proj=gate,
            up_proj=up,
            down_proj=down,
            activation=sw.activation,
        ))
        wrapped += 1
        if layer_idx in required_mixed_layers:
            seen_required.add(layer_idx)
    missing = sorted(required_mixed_layers - seen_required)
    if missing:
        raise MixedBitSwitchGLUError(
            "metadata declares mixed gate/up bits for layer(s) "
            f"{missing}, but they were not wrapped")
    return wrapped


class UnsupportedRuntimeAdapter(ValueError):
    pass


def _runtime_adapter_kind(manifest: dict) -> str:
    family = manifest.get("architecture", {}).get("family")
    required_ops = set(manifest.get("required_ops", []))
    dense_affine_ops = {
        "affine_dequant",
        "mxfp4_dequant",
        "mxfp8_dequant",
        "fp16_passthrough",
        "f32_passthrough",
    }
    qwen_kquant_ops = dense_affine_ops | {"kquant_dequant"}
    dsv4_ops = {
        "affine_dequant",
        "fp16_passthrough",
        "f32_passthrough",
        "raw_dtype_passthrough",
        "mxfp4_dequant",
        "mxfp8_dequant",
        "kquant_dequant",
        # IQ_K routed experts serve through the mlx-iqk decode kernels, which
        # read the `iqk_relayout` bundle layout only. A package still on the
        # quantizer's own wire carries the same op and is refused at install
        # by layout, where the reason can be stated.
        "iqk_dequant",
    }
    qwen4_ops = {
        "iqk_dequant",
        "kquant_dequant",
        "raw_dtype_passthrough",
    }

    if family == "deepseek_v4_flash":
        unexpected = required_ops - dsv4_ops
        if unexpected:
            raise UnsupportedRuntimeAdapter(
                "unsupported DeepSeek V4 runtime ops "
                f"{sorted(unexpected)!r}; required_ops={sorted(required_ops)!r}")
        return "mjtq_dsv4"
    if family == "qwen4_exp":
        unexpected = required_ops - qwen4_ops
        if required_ops != qwen4_ops:
            raise UnsupportedRuntimeAdapter(
                "unsupported Qwen4 runtime ops "
                f"{sorted(unexpected)!r}; required_ops={sorted(required_ops)!r}"
            )
        return "qwen4_iqk_moe"
    if family == "qwen3_5_dense" and required_ops <= dense_affine_ops:
        return "regular_jang_v2"
    if (
        family == "qwen3_5_moe"
        and "kquant_dequant" in required_ops
        and required_ops <= qwen_kquant_ops
    ):
        return "qwen_kquant_moe"

    raise UnsupportedRuntimeAdapter(
        f"unsupported runtime adapter for family={family!r}, "
        f"required_ops={sorted(required_ops)!r}")


def build_model(
    manifest: dict,
    package_dir: Path,
    *,
    load_jang_fn=None,
    load_dsv4_fn=None,
    load_qwen4_fn=None,
    load_qwen_kquant_fn=None,
    context_limit: int | None = None,
    context_limit_explicit: bool = False,
    cache_routing: str = "auto",
    cache_routing_factor: float | None = None,
    cache_routing_protected_routes: int | None = None,
):
    """Build the manifest-selected model and tokenizer from package files."""
    _silence_known_transformers_warnings()  # jang loads the tokenizer below
    package_dir = Path(package_dir)
    adapter = _runtime_adapter_kind(manifest)
    from moespresso.runtime.qwen4.cache_routing_config import resolve_cache_routing

    routing = resolve_cache_routing(
        cache_routing, factor=cache_routing_factor, protected_routes=cache_routing_protected_routes,
    )
    if routing.enabled and adapter != "qwen4_iqk_moe":
        raise UnsupportedRuntimeAdapter("cache routing is supported only by the Qwen4 pooled adapter")

    if adapter == "mjtq_dsv4":
        if load_dsv4_fn is None:
            from moespresso.runtime.deepseek_v4.model import load_deepseek_v4_package_model
            load_dsv4_fn = load_deepseek_v4_package_model
            return load_dsv4_fn(
                manifest,
                package_dir,
                context_limit=context_limit,
                context_limit_explicit=context_limit_explicit,
            )
        return load_dsv4_fn(manifest, package_dir)

    if adapter == "qwen4_iqk_moe":
        if load_qwen4_fn is None:
            from moespresso.runtime.qwen4.load import (
                load_qwen4_iqk_package_model as load_qwen4_fn,
            )
        kwargs = {}
        if context_limit is not None:
            kwargs["max_context_tokens"] = int(context_limit)
        kwargs.update(routing.load_options(include_off=True))
        return load_qwen4_fn(manifest, package_dir, **kwargs)

    if adapter == "regular_jang_v2":
        if load_jang_fn is None:
            from jang_tools.loader import load_jang_model
            load_jang_fn = load_jang_model
        return load_jang_fn(package_dir)

    index = _routed_expert_index(package_dir)
    required_mixed_layers = set() if index is None else _mixed_gate_up_layers(index)
    if load_qwen_kquant_fn is None:
        load_qwen_kquant_fn = _load_qwen_kquant_model
    model, tokenizer = load_qwen_kquant_fn(manifest, package_dir)

    jang_cfg_path = package_dir / "jang_config.json"
    jcfg = {}
    if jang_cfg_path.exists():
        with open(jang_cfg_path) as f:
            jcfg = json.load(f)
        tensor_map = jcfg.get("quantization", {}).get("tensor_map", {})
        if tensor_map:
            _apply_tensor_map(model, tensor_map)
    if index is not None:
        # Bundle packages: jang's loader cannot hydrate routed experts (no
        # stacked keys on disk); install them from the bundles, fail-loud.
        _install_routed_experts_from_bundles(
            model, package_dir, index)
    _wrap_mixed_bit_switchglus(model, required_mixed_layers=required_mixed_layers)

    if adapter == "qwen_kquant_moe":
        # The package routers are hydrated F32 linears whose values lie exactly
        # on the BF16 lattice. Install the fixed decode GEMV only after every
        # router weight is present; the wrapper retains each original linear for
        # prefill and fail-closed fallback.
        from moespresso.runtime.qwen.router_gemv import (
            install_router_bf16_f32_gemv,
        )

        install_router_bf16_f32_gemv(model)

        # Replace compatible resident K-quant SwitchGLU modules with the sorted
        # routed implementation. A layer remains unchanged when its projections
        # cannot form a compatible K-quant stack.
        from moespresso.runtime.qwen.sorted_switch_glu import (
            install_sorted_kquant_switchglus,
        )

        install_sorted_kquant_switchglus(model)

        # Coalesce long-prompt prefill work for the sorted routed-MoE path so an
        # active expert can serve more token-expert pairs per read. The default
        # uses a 4096-token chunk for long prompts; short prompts retain the
        # mlx-lm chunk and MOESPRESSO_QWEN_PREFILL_CHUNK overrides the value.
        # Chunking crosses the q8 KV dense/quantized boundary, so it is a
        # numerical variant validated with the model-family quality checks.
        from moespresso.runtime.qwen.prefill_chunk import install_prefill_chunk

        install_prefill_chunk(model)

        # Eligible full-attention prefill chunks over the q8 KV cache use the
        # mlx_kquant flash kernel, which avoids materializing the score tensor.
        from moespresso.runtime.qwen.full_attention import (
            install_flash_prefill_attention,
        )

        install_flash_prefill_attention(model)

        # The guarded decode-only fusion wraps recurrent layers and delegates
        # off-contract calls to the pinned MLX LM implementation. Its environment
        # variable disables the wrapper.
        from moespresso.runtime.qwen.gdn_decode import install_fused_gdn_decode

        install_fused_gdn_decode(model)

    return model, tokenizer
