"""DeepSeek-V4 package writer helpers."""

from __future__ import annotations

import numpy as np

from moespresso.package.bundle import assemble_layer_bundle, ds4_source_to_mxfp4_components
from moespresso.package.deepseek_v4.recipe import expert_target_from_allocation
from moespresso.package.kquant_backend import encode_kquant_weight
from moespresso.package.kquant_cache import KQuantEncodeCache, source_identity_from_arrays
from moespresso.package.kquant_format import KQUANT_GEOMETRY
from moespresso.package.tq import quantize_tq
from moespresso.probe.deepseek_v4.experts import DecodedExpertGroup


def quantize_experts_streamed(
    group: DecodedExpertGroup,
    layer: int,
    projection: str,
    bits: int,
    seed: int,
    max_experts: int | None = None,
) -> dict[str, np.ndarray]:
    """TQ-quantize DS4 decoded separate FP4 experts one expert at a time."""
    import mlx.core as mx

    experts = group.experts(layer)
    if max_experts is not None:
        experts = experts[:max_experts]
    packed_list, norms_list = [], []
    for expert_index in experts:
        decoded = group.decode(
            layer=layer,
            expert_index=expert_index,
            projection=projection,
            out_dtype=np.float32,
        )
        quantized = quantize_tq(decoded, bits, seed)
        packed_list.append(quantized["tq_packed"])
        norms_list.append(quantized["tq_norms"])
        del decoded
        mx.eval()
        mx.clear_cache()
    return {
        "tq_packed": np.stack(packed_list, axis=0),
        "tq_norms": np.stack(norms_list, axis=0),
        "tq_bits": np.array([bits], dtype=np.uint8),
    }


def bundle_row(
    group: DecodedExpertGroup,
    layer: int,
    expert_index: int,
    allocs: dict[str, dict],
    seed: int,
    *,
    kquant_imatrix_vectors: dict[str, np.ndarray] | None = None,
    kquant_encoder=None,
    kquant_expert_loader=None,
    kquant_cache: KQuantEncodeCache | None = None,
    kquant_cache_context: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Quantize one DS4 expert's gate/up/down payload into one bundle row."""
    import mlx.core as mx

    comps: dict[tuple[str, str], np.ndarray] = {}
    bits: dict[str, int] = {}
    codecs: dict[str, str] = {}
    kquant_codecs: dict[str, str] = {}
    for projection in ("gate", "up", "down"):
        alloc = allocs[projection]
        proj_key = f"{projection}_proj"
        codec = alloc.get("codec", alloc.get("format", "tq"))
        if codec == "mxfp4":
            packed_i8, scales_u8 = group.storage(
                layer=layer,
                expert_index=expert_index,
                projection=projection,
            )
            converted = ds4_source_to_mxfp4_components(packed_i8, scales_u8)
            comps[(proj_key, "packed")] = converted["packed"][None, ...]
            comps[(proj_key, "scales")] = converted["scales"][None, ...]
            bits[proj_key] = 4
            codecs[proj_key] = "mxfp4"
            del packed_i8, scales_u8, converted
        elif codec == "tq":
            decoded = group.decode(
                layer=layer,
                expert_index=expert_index,
                projection=projection,
                out_dtype=np.float32,
            )
            quantized = quantize_tq(decoded, alloc["bits"], seed)
            comps[(proj_key, "packed")] = quantized["tq_packed"][None, ...]
            comps[(proj_key, "norms")] = quantized["tq_norms"][None, ...]
            bits[proj_key] = int(alloc["bits"])
            codecs[proj_key] = "tq"
            del decoded, quantized
            mx.eval()
            mx.clear_cache()
        elif alloc.get("format") == "kquant" or codec == "kquant":
            if kquant_imatrix_vectors is None and kquant_expert_loader is None:
                raise ValueError(
                    f"K-quant DS4 expert codec requires imatrix vectors for "
                    f"layer={layer} projection={projection}")
            target = expert_target_from_allocation(alloc)
            encoder = kquant_encoder or encode_kquant_weight
            encoded = None
            metadata = None
            if kquant_expert_loader is not None:
                encoded = kquant_expert_loader(target, int(expert_index))
            else:
                packed_i8, scales_u8 = group.storage(
                    layer=layer,
                    expert_index=expert_index,
                    projection=projection,
                )
                if kquant_cache is not None:
                    metadata = kquant_cache.metadata_for(
                        source=source_identity_from_arrays(
                            "deepseek_v4_fp4_expert_storage",
                            {
                                "weight": packed_i8,
                                "scale": scales_u8,
                            },
                            layer_index=int(layer),
                            expert_index=int(expert_index),
                            projection=projection,
                        ),
                        target=target,
                        imatrix_vectors=kquant_imatrix_vectors or {},
                        context=kquant_cache_context,
                    )
                    encoded = kquant_cache.get(metadata)
            if encoded is None:
                decoded = group.decode(
                    layer=layer,
                    expert_index=expert_index,
                    projection=projection,
                    out_dtype=np.float32,
                )
                encoded = encoder(
                    decoded.astype(np.float32),
                    target,
                    kquant_imatrix_vectors or {},
                )
                if metadata is not None:
                    kquant_cache.put(metadata, encoded)
                del decoded
            if encoded.codec != target.codec:
                raise ValueError(
                    f"K-quant encoder returned codec {encoded.codec!r}, expected "
                    f"{target.codec!r} for layer={layer} projection={projection}")
            comps[(proj_key, "weight")] = encoded.weight[None, ...]
            comps[(proj_key, "scales")] = encoded.scales[None, ...]
            bits[proj_key] = int(KQUANT_GEOMETRY[target.codec].bits)
            codecs[proj_key] = "kquant"
            kquant_codecs[proj_key] = target.codec
            del encoded
            mx.eval()
            mx.clear_cache()
        else:
            raise ValueError(
                f"unsupported DS4 expert codec {codec!r} for layer={layer} "
                f"projection={projection}")
    if kquant_codecs:
        bundle, geometry = assemble_layer_bundle(
            comps,
            bits,
            codecs=codecs,
            kquant_codecs=kquant_codecs,
        )
    else:
        bundle, geometry = assemble_layer_bundle(comps, bits, codecs=codecs)
    return np.ascontiguousarray(bundle[0]), geometry
