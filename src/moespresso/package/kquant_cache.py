"""Content-addressed cache for encoded K-quant wire tensors.

This cache accelerates diagnostic iteration while preserving every input that
affects model quality. Keys include source bytes, codec, imatrix, target identity,
and caller context so stale encoded tensors fail to match instead of silently drifting.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from moespresso.package.deepseek_v4.recipe import DS4KQuantDenseTarget, DS4KQuantExpertTarget
from moespresso.package.kquant_backend import KQuantEncodedWeight
from moespresso.package.qwen.recipe import QwenKQuantDenseTarget, QwenKQuantExpertTarget


class KQuantCacheError(RuntimeError):
    pass


def _canonical_json(payload: dict) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _array_identity(array) -> dict:
    arr = np.ascontiguousarray(array)
    return {
        "dtype": str(arr.dtype),
        "shape": [int(v) for v in arr.shape],
        "sha256": _sha256_bytes(arr.tobytes()),
    }


KQuantCacheTarget = (
    DS4KQuantExpertTarget
    | DS4KQuantDenseTarget
    | QwenKQuantExpertTarget
    | QwenKQuantDenseTarget
)


def _target_identity(target: KQuantCacheTarget) -> dict:
    base = {
        "codec": target.codec,
        "gguf_tensor": target.gguf_tensor,
        "imatrix_key": target.imatrix_key,
        "module_path": target.module_path,
        "module_weight_key": target.module_weight_key,
    }
    if isinstance(target, DS4KQuantExpertTarget):
        base.update({
            "kind": "expert",
            "layer_index": int(target.layer_index),
            "projection": target.projection,
            "source_weight_template": target.source_weight_template,
            "source_scale_template": target.source_scale_template,
        })
    elif isinstance(target, QwenKQuantExpertTarget):
        base.update({
            "kind": "expert",
            "layer_index": int(target.layer_index),
            "projection": target.projection,
            "source_name": target.source_name,
            "source_projection": target.source_projection,
        })
    else:
        base.update({
            "kind": "dense",
            "source_name": target.source_name,
            "role": target.role,
            "layer_index": target.layer_index,
            "requires_imatrix": bool(getattr(target, "requires_imatrix", True)),
        })
    return base


def _imatrix_identity(
    target: KQuantCacheTarget,
    imatrix_vectors: dict[str, np.ndarray],
) -> dict:
    vec = imatrix_vectors.get(target.imatrix_key)
    if vec is None:
        return {"key": target.imatrix_key, "present": False}
    return {"key": target.imatrix_key, "present": True, **_array_identity(vec)}


class KQuantEncodeCache:
    """Disk cache for already-encoded `KQuantEncodedWeight` values."""

    schema = "moespresso.kquant_encode_cache.v1"

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        self.writes = 0

    def metadata_for(
        self,
        *,
        source: dict,
        target: KQuantCacheTarget,
        imatrix_vectors: dict[str, np.ndarray],
        context: dict | None = None,
    ) -> dict:
        payload = {
            "schema": self.schema,
            "source": source,
            "target": _target_identity(target),
            "imatrix": _imatrix_identity(target, imatrix_vectors),
            "context": context or {},
        }
        key = _sha256_bytes(_canonical_json(payload))
        return {**payload, "key": key}

    def get(self, metadata: dict) -> KQuantEncodedWeight | None:
        path = self.root / f"{metadata['key']}.npz"
        if not path.exists():
            self.misses += 1
            return None
        try:
            with np.load(path, allow_pickle=False) as loaded:
                stored_meta = bytes(loaded["metadata_json"].tolist()).decode("utf-8")
                stored = json.loads(stored_meta)
                weight = np.ascontiguousarray(loaded["weight"])
                scales = np.ascontiguousarray(loaded["scales"])
        except Exception as exc:
            raise KQuantCacheError(f"failed to read K-quant cache entry {path}") from exc

        if stored.get("key") != metadata["key"]:
            raise KQuantCacheError(
                f"K-quant cache entry {path} has key {stored.get('key')!r}, "
                f"expected {metadata['key']!r}")
        if stored.get("payload") != metadata:
            raise KQuantCacheError(f"K-quant cache metadata mismatch for {path}")
        if weight.dtype != np.uint8 or scales.dtype != np.uint8:
            raise KQuantCacheError(f"K-quant cache entry {path} is not uint8 wire data")
        encoded_meta = stored.get("encoded", {})
        if _array_identity(weight) != encoded_meta.get("weight"):
            raise KQuantCacheError(f"K-quant cache weight checksum mismatch for {path}")
        if _array_identity(scales) != encoded_meta.get("scales"):
            raise KQuantCacheError(f"K-quant cache scales checksum mismatch for {path}")

        self.hits += 1
        codec = metadata["target"]["codec"]
        return KQuantEncodedWeight(codec=codec, weight=weight, scales=scales)

    def put(self, metadata: dict, encoded: KQuantEncodedWeight) -> None:
        if encoded.codec != metadata["target"]["codec"]:
            raise KQuantCacheError(
                f"refusing to cache codec {encoded.codec!r} for target "
                f"{metadata['target']['codec']!r}")
        path = self.root / f"{metadata['key']}.npz"
        tmp = self.root / f"{metadata['key']}.tmp.npz"
        weight = np.ascontiguousarray(encoded.weight, dtype=np.uint8)
        scales = np.ascontiguousarray(encoded.scales, dtype=np.uint8)
        stored = {
            "key": metadata["key"],
            "payload": metadata,
            "encoded": {
                "weight": _array_identity(weight),
                "scales": _array_identity(scales),
            },
        }
        meta_json = np.frombuffer(_canonical_json(stored), dtype=np.uint8)
        with open(tmp, "wb") as f:
            np.savez(f, metadata_json=meta_json, weight=weight, scales=scales)
        tmp.replace(path)
        self.writes += 1

    def summary(self) -> dict:
        return {
            "enabled": True,
            "hits": int(self.hits),
            "misses": int(self.misses),
            "writes": int(self.writes),
        }


def source_identity_from_arrays(kind: str, arrays: dict[str, np.ndarray], **fields) -> dict:
    return {
        "kind": kind,
        **fields,
        "arrays": {
            key: _array_identity(value)
            for key, value in sorted(arrays.items())
        },
    }
