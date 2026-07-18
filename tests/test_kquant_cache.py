from __future__ import annotations

import numpy as np
import pytest

from moespresso.package.kquant_backend import KQuantEncodedWeight
from moespresso.package.kquant_cache import (
    KQuantCacheError,
    KQuantEncodeCache,
    source_identity_from_arrays,
)
from moespresso.package.deepseek_v4.recipe import DS4KQuantExpertTarget


def _target(codec="q2_k"):
    return DS4KQuantExpertTarget(
        layer_index=7,
        projection="down",
        codec=codec,
        gguf_tensor="blk.7.ffn_down_exps.weight",
        imatrix_key="blk.7.ffn_down_exps.weight",
        source_weight_template="layers.7.ffn.experts.{expert}.w2.weight",
        source_scale_template="layers.7.ffn.experts.{expert}.w2.scale",
        module_path="model.layers.7.mlp.switch_mlp.down_proj",
        module_weight_key="model.layers.7.mlp.switch_mlp.down_proj.weight",
    )


def _encoded(value=5):
    return KQuantEncodedWeight(
        codec="q2_k",
        weight=np.full((2, 84), value, dtype=np.uint8),
        scales=np.zeros((1,), dtype=np.uint8),
    )


def test_kquant_cache_reuses_matching_source_codec_imatrix_and_context(tmp_path):
    cache = KQuantEncodeCache(tmp_path)
    target = _target()
    imatrix = {"blk.7.ffn_down_exps.weight": np.ones(256, dtype=np.float32)}
    source = source_identity_from_arrays(
        "deepseek_v4_fp4_expert_storage",
        {
            "weight": np.arange(64, dtype=np.int8),
            "scale": np.arange(2, dtype=np.uint8),
        },
        layer_index=7,
        expert_index=3,
        projection="down",
    )
    metadata = cache.metadata_for(
        source=source,
        target=target,
        imatrix_vectors=imatrix,
        context={"recipe_mode": "faithful_recipe"},
    )

    assert cache.get(metadata) is None
    cache.put(metadata, _encoded())
    cached = cache.get(metadata)

    assert cached is not None
    assert cached.codec == "q2_k"
    np.testing.assert_array_equal(cached.weight, np.full((2, 84), 5, dtype=np.uint8))
    assert "path" not in cache.summary()
    assert cache.summary()["enabled"] is True
    assert cache.summary()["hits"] == 1
    assert cache.summary()["misses"] == 1
    assert cache.summary()["writes"] == 1


def test_kquant_cache_key_changes_with_source_codec_imatrix_or_diagnostic_mode(tmp_path):
    cache = KQuantEncodeCache(tmp_path)
    target = _target()
    imatrix = {"blk.7.ffn_down_exps.weight": np.ones(256, dtype=np.float32)}
    source = source_identity_from_arrays(
        "dense_matrix",
        {"weight": np.ones((2, 256), dtype=np.float32)},
        source_name="x.weight",
    )
    base = cache.metadata_for(
        source=source,
        target=target,
        imatrix_vectors=imatrix,
        context={"recipe_mode": "faithful_recipe"},
    )
    changed_source = cache.metadata_for(
        source=source_identity_from_arrays(
            "dense_matrix",
            {"weight": np.full((2, 256), 2, dtype=np.float32)},
            source_name="x.weight",
        ),
        target=target,
        imatrix_vectors=imatrix,
        context={"recipe_mode": "faithful_recipe"},
    )
    changed_codec = cache.metadata_for(
        source=source,
        target=_target("q3_k"),
        imatrix_vectors=imatrix,
        context={"recipe_mode": "faithful_recipe"},
    )
    changed_imatrix = cache.metadata_for(
        source=source,
        target=target,
        imatrix_vectors={"blk.7.ffn_down_exps.weight": np.full(256, 2, dtype=np.float32)},
        context={"recipe_mode": "faithful_recipe"},
    )
    changed_mode = cache.metadata_for(
        source=source,
        target=target,
        imatrix_vectors=imatrix,
        context={"recipe_mode": "fast_diagnostic"},
    )

    keys = {
        base["key"],
        changed_source["key"],
        changed_codec["key"],
        changed_imatrix["key"],
        changed_mode["key"],
    }
    assert len(keys) == 5


def test_kquant_cache_corruption_fails_closed(tmp_path):
    cache = KQuantEncodeCache(tmp_path)
    target = _target()
    imatrix = {"blk.7.ffn_down_exps.weight": np.ones(256, dtype=np.float32)}
    metadata = cache.metadata_for(
        source=source_identity_from_arrays(
            "dense_matrix",
            {"weight": np.ones((2, 256), dtype=np.float32)},
            source_name="x.weight",
        ),
        target=target,
        imatrix_vectors=imatrix,
        context={"recipe_mode": "faithful_recipe"},
    )
    cache.put(metadata, _encoded())
    (tmp_path / f"{metadata['key']}.npz").write_bytes(b"not a valid npz")

    with pytest.raises(KQuantCacheError, match="failed to read"):
        cache.get(metadata)
