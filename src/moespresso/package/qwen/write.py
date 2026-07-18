"""Qwen package writer helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from moespresso.package.kquant_backend import encode_kquant_weight
from moespresso.package.kquant_cache import KQuantEncodeCache, source_identity_from_arrays
from moespresso.package.qwen.recipe import expert_submatrix, expert_target_from_allocation
from moespresso.probe import weight_io


def encode_kquant_experts_streamed(
    model_dir: Path,
    header,
    alloc: dict,
    *,
    max_experts: int | None = None,
    kquant_imatrix_vectors: dict[str, np.ndarray] | None = None,
    kquant_encoder=None,
    kquant_expert_loader=None,
    kquant_cache: KQuantEncodeCache | None = None,
    kquant_cache_context: dict | None = None,
):
    """Encode a stacked Qwen expert projection into per-expert K-quant wire rows.

    When `kquant_expert_loader` is set, each expert row comes from the loader
    (for example byte-faithful GGUF reads) and the source decode, encoder, and
    encode cache are bypassed.
    """
    target = expert_target_from_allocation(alloc)
    if kquant_expert_loader is not None:
        num_experts = int(header.shape[0])
        if max_experts is not None:
            num_experts = min(num_experts, int(max_experts))
        loaded = []
        for expert_index in range(num_experts):
            encoded = kquant_expert_loader(target, expert_index)
            if encoded.codec != target.codec:
                raise ValueError(
                    f"K-quant expert loader returned codec {encoded.codec!r}, "
                    f"expected {target.codec!r} for layer={target.layer_index} "
                    f"projection={target.projection}")
            loaded.append(encoded)
        return loaded

    import mlx.core as mx

    encoder = kquant_encoder or encode_kquant_weight
    imatrix_vectors = kquant_imatrix_vectors or {}
    encoded_list = []
    for expert_index, expert in enumerate(
        weight_io.iter_experts(model_dir, header, max_experts=max_experts)
    ):
        sub = np.ascontiguousarray(
            expert_submatrix(expert, target),
            dtype=np.float32,
        )
        encoded = None
        metadata = None
        if kquant_cache is not None:
            metadata = kquant_cache.metadata_for(
                source=source_identity_from_arrays(
                    "qwen_stacked_expert",
                    {"weight": sub},
                    source_name=target.source_name,
                    layer_index=int(target.layer_index),
                    expert_index=int(expert_index),
                    projection=target.projection,
                    source_projection=target.source_projection,
                ),
                target=target,
                imatrix_vectors=imatrix_vectors,
                context=kquant_cache_context,
            )
            encoded = kquant_cache.get(metadata)
        if encoded is None:
            encoded = encoder(sub, target, imatrix_vectors)
            if metadata is not None:
                kquant_cache.put(metadata, encoded)
        if encoded.codec != target.codec:
            raise ValueError(
                f"K-quant encoder returned codec {encoded.codec!r}, expected "
                f"{target.codec!r} for layer={target.layer_index} "
                f"projection={target.projection}")
        encoded_list.append(encoded)
        del expert, sub, encoded
        mx.eval()
        mx.clear_cache()
    return encoded_list
