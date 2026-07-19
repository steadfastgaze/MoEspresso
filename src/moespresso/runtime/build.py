"""Manifest-driven model build for mjtq serve along the established path.

mjtq reuses the serve path already validated on Qwen3.5/3.6 mixed-affine+TQ:
jang's `load_jangtq_model` builds the graph from the package's `config.json` (affine
non-experts as mlx QuantizedLinear via the per-module `quantization` block; TQ experts
as TurboQuantSwitchLinear metal-kernel modules), then a per-tensor `tensor_map`
override fixes mixed-precision bits/group_size. No dequant at load: TQ weights stay
packed and the GPU kernel runs them; mlx affine modules dequant in-layer at inference.

The package carries `config.json` + `jang_config.json` as jang-compatible sidecars
generated from the manifest at convert time (a compat view; the manifest stays the
source of truth and avoids source archaeology). This module just feeds them to the
jang loader. No numpy on this path (jang's hot path is its metal kernel + mlx).

Needs the standard runtime dependencies (mlx, mlx-lm, jang). Imports are lazy.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import sys
from pathlib import Path

# transformers (5.8.x/5.9.x) falsely fires a "fix_mistral_regex" warning for non-Mistral
# tokenizers loaded from a dir that also holds model files (the mjtq package): an
# upstream bug (huggingface/transformers#42591, fixed by #45444 but not in this line).
# The tokenizer is correct (loads as Qwen2Tokenizer, round-trips exactly), so drop
# only this one false message; never set fix_mistral_regex=True (it changes a correct
# regex and is reported to break non-Mistral tokenization).
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


def _load_jangtq_quietly(load_fn, package_dir):
    """Call jang's loader, discarding its verbose stdout banner on success.

    jang's load_jangtq_model prints a multi-line load banner to stdout (Loading JANGTQ,
    seed/bits_map (where bits_map is the deliberate sentinel 0, see package/sidecars),
    Replaced N modules, [warmup]..., Done). It is third-party (cannot be edited), so
    capture that stdout and drop it on a successful load. On failure the captured text
    is re-emitted first, so a broken load stays diagnosable. Only this call's stdout is
    touched.
    """
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            return load_fn(package_dir)
    except BaseException:
        sys.stdout.write(buf.getvalue())  # surface jang's progress so the failure is readable
        sys.stdout.flush()
        raise


def _apply_tensor_map(model, tensor_map: dict) -> None:
    """Override bits/group_size on QuantizedLinear modules from the explicit map.

    Mixed per-tensor affine: jang builds modules from config.json's quantization
    block; this pins each module's exact bits/group_size so shape-guessing cannot
    pick the wrong precision (the fix for mixed-affine)."""
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
    """Return layers whose routed gate/up TQ bits differ, using headers only."""
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


def _install_routed_experts_from_bundles(model, package_dir: Path, index,
                                         *, seed: int) -> int:
    """Install resident expert modules from the bundle tensors.

    jang's loader hydrates TurboQuantSwitchLinear modules from the pre-bundle
    stacked keys; bundle packages carry none, so the loader silently leaves the
    plain (random-init) SwitchGLU in place. This pass owns the routed payload
    for the resident (non-streaming) serve path: per layer it reads the bundle
    once, splits components per the metadata geometry, and replaces each
    projection with a TurboQuantSwitchLinear carrying the exact packed/norms
    bytes. Anything missing fails loudly (never a quietly wrong model).
    """
    import mlx.core as mx
    from moespresso.package.bundle import KQUANT_CODEC, TQ_CODEC

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
            if geo.codec == TQ_CODEC:
                from jang_tools.turboquant.tq_kernel import TurboQuantSwitchLinear

                in_features = geo.packed_cols * (32 // geo.bits)
                mod = TurboQuantSwitchLinear(
                    in_features, geo.out_features, n_exp, bits=geo.bits, seed=seed)
                weight_component = "packed"
            elif geo.codec == KQUANT_CODEC:
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
            weight_dtype = mx.uint8 if geo.codec == KQUANT_CODEC else mx.uint32
            weight = mx.zeros(
                (n_exp, geo.out_features, geo.packed_cols),
                dtype=weight_dtype,
            )
            if geo.codec == TQ_CODEC:
                norms = mx.zeros((n_exp, geo.out_features), dtype=mx.float16)
                mx.eval(weight, norms)
                norms_view = memoryview(norms).cast("B")
                nc = comps[(proj, "norms")]
                nn = nc["nbytes"]
            else:
                norms = None
                norms_view = None
                nc = None
                nn = 0
            weight_view = memoryview(weight).cast("B")
            wc = comps[(proj, weight_component)]
            wn = wc["nbytes"]
            for e in range(n_exp):
                row = rows[e * row_bytes:(e + 1) * row_bytes]
                weight_view[e * wn:(e + 1) * wn] = (
                    row[wc["offset"]:wc["offset"] + wn])
                if norms_view is not None:
                    norms_view[e * nn:(e + 1) * nn] = (
                        row[nc["offset"]:nc["offset"] + nn])
            if geo.codec == TQ_CODEC:
                mod.packed = weight
                mod.norms = norms
            else:
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
    tokenizer = load_tokenizer_fn(package_dir)
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
        "tq_dequant",
        "mxfp4_dequant",
        "mxfp8_dequant",
        "kquant_dequant",
    }

    if family == "deepseek_v4_flash":
        unexpected = required_ops - dsv4_ops
        if unexpected:
            raise UnsupportedRuntimeAdapter(
                "unsupported DeepSeek V4 runtime ops "
                f"{sorted(unexpected)!r}; required_ops={sorted(required_ops)!r}")
        return "mjtq_dsv4"
    if family == "qwen3_5_dense" and required_ops <= dense_affine_ops:
        return "regular_jang_v2"
    if (
        family == "qwen3_5_moe"
        and "kquant_dequant" in required_ops
        and "tq_dequant" not in required_ops
        and required_ops <= qwen_kquant_ops
    ):
        return "qwen_kquant_moe"
    if "tq_dequant" in required_ops and family != "qwen3_5_dense":
        return "jangtq_moe"

    raise UnsupportedRuntimeAdapter(
        f"unsupported runtime adapter for family={family!r}, "
        f"required_ops={sorted(required_ops)!r}")


def build_model(
    manifest: dict,
    package_dir: Path,
    *,
    load_jangtq_fn=None,
    load_jang_fn=None,
    load_dsv4_fn=None,
    load_qwen_kquant_fn=None,
):
    """Build (model, tokenizer) from a mjtq package via the jang loader.

    Reads the jang-compatible sidecars in the package dir (config.json,
    jang_config.json) that convert generated from the manifest. Returns
    (model, tokenizer): the tokenizer is the one jang's loader produced (mlx_lm
    load_tokenizer + eos/chat handling), the same one the established path uses;
    it is not re-loaded separately (that would diverge from the established path)."""
    _silence_known_transformers_warnings()  # jang loads the tokenizer below
    package_dir = Path(package_dir)
    adapter = _runtime_adapter_kind(manifest)

    if adapter == "mjtq_dsv4":
        if load_dsv4_fn is None:
            from moespresso.runtime.deepseek_v4.model import load_deepseek_v4_package_model
            load_dsv4_fn = load_deepseek_v4_package_model
        return load_dsv4_fn(manifest, package_dir)

    if adapter == "regular_jang_v2":
        if load_jang_fn is None:
            from jang_tools.loader import load_jang_model
            load_jang_fn = load_jang_model
        return load_jang_fn(package_dir)

    index = _routed_expert_index(package_dir)
    required_mixed_layers = set() if index is None else _mixed_gate_up_layers(index)
    if adapter == "qwen_kquant_moe":
        if load_qwen_kquant_fn is None:
            load_qwen_kquant_fn = _load_qwen_kquant_model
        model, tokenizer = load_qwen_kquant_fn(manifest, package_dir)
    else:
        if load_jangtq_fn is None:
            from jang_tools.load_jangtq import load_jangtq_model
            load_jangtq_fn = load_jangtq_model
        model, tokenizer = _load_jangtq_quietly(load_jangtq_fn, package_dir)

    jang_cfg_path = package_dir / "jang_config.json"
    jcfg = {}
    if jang_cfg_path.exists():
        with open(jang_cfg_path) as f:
            jcfg = json.load(f)
        tensor_map = jcfg.get("quantization", {}).get("tensor_map", {})
        if tensor_map:
            _apply_tensor_map(model, tensor_map)
    if index is not None:
        # bundle packages: jang's loader cannot hydrate routed experts (no
        # stacked keys on disk); install them from the bundles, fail-loud.
        _install_routed_experts_from_bundles(
            model, package_dir, index, seed=int(jcfg.get("mxtq_seed", 42)))
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

        # First optimized MoE path: swap the resident K-quant SwitchGLU seams
        # for the sorted routed route (sorted-ids prefill GEMM over the full
        # expert stacks). Fail-closed per layer and behind
        # MOESPRESSO_QWEN_MOE_SORTED; a no-op when the kill switch is off or a
        # layer's projections are not combinable K-quant stacks.
        from moespresso.runtime.qwen.sorted_switch_glu import (
            install_sorted_kquant_switchglus,
        )

        install_sorted_kquant_switchglus(model)

        # Served prefill chunk coalescing for the sorted routed-MoE path. A
        # larger prompt chunk reads each active expert's weights fewer times
        # across a long prefill. Under the composed head-dimension-256 prefill
        # attention this did not convert to served throughput (that path
        # materializes a quadratic score tensor per chunk that grew faster than
        # the routed-MoE saving, so peak climbed and the rate was flat at 4096
        # and worse beyond it). The flash prefill route removes the score-tensor
        # materialization and the re-pricing reverses that verdict: at 37K under
        # the flash route chunk 4096 runs about 10 t/s faster than chunk 2048 and
        # cuts about 0.6 s off the time to first token, at a 26.43 GiB peak that
        # fits the 32 GB budget. So the default raises the chunk to 4096 for long
        # prompts (short prompts keep the mlx_lm chunk);
        # MOESPRESSO_QWEN_PREFILL_CHUNK=<n> overrides the value. Chunk 8192 stays
        # out (over budget and slower). Math-affecting through the q8 KV
        # dense/quantized boundary: chunk 4096 forks the 37K greedy stream from
        # chunk 2048, so the change is judged by the quality ladder (gate stays
        # 9/9 clean-pass). Token identity is not required for this numerical variant.
        from moespresso.runtime.qwen.prefill_chunk import install_prefill_chunk

        install_prefill_chunk(model)

        # Flash D=256 prefill attention for the full-attention layers: eligible
        # prefill chunks over the q8 KV cache dispatch to the mlx_kquant flash
        # kernel with no materialized score tensor. The float32 memory-lever
        # form passed the full quality ladder (engaged prefill NLL improves,
        # gate clean, A/A rail recorded) and cuts the 37K served peak by
        # 4.16 GB at a 3.6% TTFT cost, so it is the default.
        # MOESPRESSO_QWEN_PREFILL_FLASH_D256=0 is the kill switch: nothing is
        # installed and serving is the stock composed path on its own rail.
        from moespresso.runtime.qwen.full_attention import (
            install_flash_prefill_attention,
        )

        install_flash_prefill_attention(model)

        # Decode-only gated-delta convolution-state fusion. The guarded route
        # wraps only the recurrent layers and delegates every off-contract call
        # to the pinned MLX LM module unchanged. Its environment variable is a
        # kill switch.
        from moespresso.runtime.qwen.gdn_decode import install_fused_gdn_decode

        install_fused_gdn_decode(model)

    return model, tokenizer
