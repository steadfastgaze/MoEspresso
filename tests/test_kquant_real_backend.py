from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from moespresso.package.kquant_backend import kquant_roundtrip_relative_error
from moespresso.package.deepseek_v4.recipe import (
    DS4KQuantDenseTarget,
    DS4KQuantExpertTarget,
    build_ds4_expert_kquant_targets,
)
from moespresso.package.kquant_recipe import (
    read_gguf_kquant_recipe,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("MOESPRESSO_RUN_KQUANT_REAL_BACKEND") != "1",
    reason="explicit real mlx-kquant backend diagnostic only",
)


def _expert_target(projection: str, codec: str) -> DS4KQuantExpertTarget:
    gguf = f"blk.0.ffn_{projection}_exps.weight"
    return DS4KQuantExpertTarget(
        layer_index=0,
        projection=projection,
        codec=codec,
        gguf_tensor=gguf,
        imatrix_key=gguf,
        source_weight_template=f"layers.0.ffn.experts.{{expert}}.{projection}.weight",
        source_scale_template=f"layers.0.ffn.experts.{{expert}}.{projection}.scale",
        module_path=f"model.layers.0.mlp.switch_mlp.{projection}_proj",
        module_weight_key=f"model.layers.0.mlp.switch_mlp.{projection}_proj.weight",
    )


def test_real_mlx_kquant_backend_roundtrips_moespresso_targets():
    kq = pytest.importorskip("mlx_kquant")
    assert kq.metallib_loads()

    rng = np.random.default_rng(7)
    weight = rng.standard_normal((4, 256), dtype=np.float32)
    imatrix = {
        "blk.0.ffn_down_exps.weight": np.linspace(
            0.5, 1.5, 256, dtype=np.float32),
        "blk.0.ffn_gate_exps.weight": np.ones(256, dtype=np.float32),
    }

    encoded_q2, err_q2 = kquant_roundtrip_relative_error(
        weight,
        _expert_target("down", "q2_k"),
        imatrix,
        stream="cpu",
    )
    encoded_iq2, err_iq2 = kquant_roundtrip_relative_error(
        weight,
        _expert_target("gate", "iq2_xxs"),
        imatrix,
        stream="cpu",
    )
    encoded_q8, err_q8 = kquant_roundtrip_relative_error(
        weight,
        DS4KQuantDenseTarget(
            source_name="layers.0.attn.wq_a.weight",
            role="attn.wq_a",
            layer_index=0,
            codec="q8_0",
            gguf_tensor="blk.0.attn_q_a.weight",
            imatrix_key="blk.0.attn_q_a.weight",
            module_path="model.layers.0.self_attn.wq_a",
            module_weight_key="model.layers.0.self_attn.wq_a.weight",
        ),
        {},
        stream="cpu",
    )

    assert encoded_q2.weight.shape == (4, 84)
    assert encoded_iq2.weight.shape == (4, 66)
    assert encoded_q8.weight.shape == (4, 272)
    assert encoded_q2.scales.tolist() == [0]
    assert encoded_iq2.scales.tolist() == [0]
    assert encoded_q8.scales.tolist() == [0]
    assert err_q8 < 0.02
    assert err_q2 < 0.35
    assert err_iq2 < 0.45


def _required_path(env_name: str) -> Path:
    value = os.environ.get(env_name)
    if not value:
        pytest.skip(f"set {env_name}")
    path = Path(value).expanduser()
    if not path.exists():
        pytest.skip(f"{env_name} not found: {path}")
    return path


def test_real_ds4_source_expert_rows_roundtrip_with_recipe_codecs():
    if os.environ.get("MOESPRESSO_RUN_DS4_KQUANT_REAL_SOURCE") != "1":
        pytest.skip("set MOESPRESSO_RUN_DS4_KQUANT_REAL_SOURCE=1")

    from moespresso.inventory.architecture_profile import family_of
    from moespresso.inventory.build import build_inventory
    from moespresso.package.deepseek_v4.kquant import load_ds4_kquant_imatrix_vectors
    from moespresso.probe.deepseek_v4.experts import DecodedExpertGroup
    from moespresso.package.convert import _layer_types, _read_config

    source = _required_path("MOESPRESSO_DEEPSEEK_V4_SOURCE")
    recipe_path = _required_path("MOESPRESSO_DS4_KQUANT_GGUF_RECIPE")
    imatrix_path = _required_path("MOESPRESSO_DEEPSEEK_V4_IMATRIX")

    config = _read_config(source)
    family = family_of(config)
    assert family == "deepseek_v4_flash"
    imatrix = load_ds4_kquant_imatrix_vectors(imatrix_path)
    inventory = build_inventory(
        source,
        layer_types=_layer_types(config),
        imatrix_keys=set(imatrix),
        family=family,
    )
    assert inventory["status"] == "valid"
    group = DecodedExpertGroup.from_inventory(inventory, source)
    layer = group.layers()[0]
    expert_index = group.experts(layer)[0]
    targets = build_ds4_expert_kquant_targets(
        read_gguf_kquant_recipe(recipe_path),
        required_layers=[layer],
    )
    by_projection = {target.projection: target for target in targets}

    down = group.decode(
        layer=layer,
        expert_index=expert_index,
        projection="down",
        out_dtype=np.float32,
    )[:8]
    gate = group.decode(
        layer=layer,
        expert_index=expert_index,
        projection="gate",
        out_dtype=np.float32,
    )[:8]

    encoded_down, err_down = kquant_roundtrip_relative_error(
        down,
        by_projection["down"],
        imatrix,
        stream="cpu",
    )
    encoded_gate, err_gate = kquant_roundtrip_relative_error(
        gate,
        by_projection["gate"],
        imatrix,
        stream="cpu",
    )

    assert encoded_down.codec == "q2_k"
    assert encoded_gate.codec == "iq2_xxs"
    assert encoded_down.weight.shape == (8, 8 * 84)
    assert encoded_gate.weight.shape == (8, 16 * 66)
    assert err_down < 0.40
    assert err_gate < 0.50
