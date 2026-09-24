"""Released Qwen4 inventory roles and source partition."""

from __future__ import annotations

from moespresso.inventory.build import build_inventory_from_headers
from moespresso.inventory.qwen4.static import (
    EXPECTED_MTP_TENSOR_COUNT,
    EXPECTED_PLE_PROVIDER_TENSOR_COUNT,
    EXPECTED_TEXT_TENSOR_COUNT,
    EXPECTED_VISION_TENSOR_COUNT,
    expected_qwen38_flash_next_header_specs,
)
from moespresso.inventory.safetensors_header import TensorHeader


_SUBJECT = {"source_root": "qwen38-flash-next", "source_format": "hf_safetensors"}


def _header(name: str, dtype: str, shape: tuple[int, ...]) -> TensorHeader:
    return TensorHeader(
        name=name,
        dtype=dtype,
        shape=shape,
        shard="model-00001-of-00131.safetensors",
    )


def _released_served_headers() -> list[TensorHeader]:
    return [
        _header(name, dtype, shape)
        for name, (dtype, shape) in expected_qwen38_flash_next_header_specs().items()
    ]


def _explicitly_omitted_headers() -> list[TensorHeader]:
    vision = [
        _header(f"model.visual.synthetic.{index}", "BF16", (1,))
        for index in range(EXPECTED_VISION_TENSOR_COUNT)
    ]
    mtp = [
        _header(f"mtp.synthetic.{index}", "BF16", (1,))
        for index in range(EXPECTED_MTP_TENSOR_COUNT)
    ]
    return vision + mtp


def _inventory(headers: list[TensorHeader], family: str = "qwen4_exp") -> dict:
    return build_inventory_from_headers(
        headers,
        _SUBJECT,
        layer_types=None,
        family=family,
    )


def test_qwen4_inventory_resolves_exact_released_text_and_provider_partition() -> None:
    headers = _released_served_headers() + _explicitly_omitted_headers()
    inventory = _inventory(headers)

    assert len(headers) == 1658
    assert inventory["status"] == "valid"
    assert inventory["family"] == "qwen4_exp"
    assert inventory["required_features"] == ["qwen4_ple_provider_inventory"]
    assert inventory["counts"] == {
        "expert": 96,
        "affine": 774,
        "expert_source": 0,
        "codec_scale": 0,
        "passthrough": 293,
        "provider": EXPECTED_PLE_PROVIDER_TENSOR_COUNT,
        "unknown": 0,
        "total": EXPECTED_TEXT_TENSOR_COUNT + EXPECTED_PLE_PROVIDER_TENSOR_COUNT,
    }
    names = {entry["source_name"] for entry in inventory["tensors"]}
    assert not any(name.startswith("model.visual.") for name in names)
    assert not any(name.startswith("mtp.") for name in names)


def test_qwen4_inventory_records_graph_and_provider_roles_without_gguf_guessing() -> None:
    inventory = _inventory(_released_served_headers())
    by_name = {entry["source_name"]: entry for entry in inventory["tensors"]}

    expert = by_name["model.language_model.layers.17.mlp.experts.gate_up_proj"]
    assert expert["kind"] == "expert"
    assert expert["role"] == "moe.expert.gate_up"
    assert expert["projection"] == "gate_up"
    assert expert["module_path"] == "layers.17.mlp.experts"
    assert "module_weight_key" not in expert

    assert by_name["model.language_model.layers.0.linear_attn.A_log"]["role"] == "gdn.A_log"
    assert (
        by_name["model.language_model.layers.3.self_attn.indexer.index_qk_proj.weight"]["role"]
        == "qsa.indexer.index_qk_proj"
    )
    assert (
        by_name["model.language_model.layers.4.attn_hyper_connection.block_inject_weight.weight"][
            "role"
        ]
        == "gr.attention.block_inject_weight"
    )
    assert by_name["model.language_model.embed_tokens.weight"]["module_weight_key"] == (
        "embedding.weight"
    )
    assert by_name["lm_head.weight"]["module_weight_key"] == "lm_head.weight"
    assert by_name[
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
    ]["module_weight_key"] == "layers.0.mixer.module.in_proj_qkv.weight"
    assert by_name["model.language_model.layers.0.linear_attn.A_log"][
        "module_weight_key"
    ] == "layers.0.mixer.module.A_log"
    assert by_name[
        "model.language_model.layers.3.self_attn.indexer.index_qk_proj.weight"
    ]["module_weight_key"] == "layers.3.mixer.module.indexer.index_qk_proj.weight"
    assert by_name[
        "model.language_model.layers.4.attn_hyper_connection.block_inject_weight.weight"
    ]["module_weight_key"] == "layers.4.attention_residual.block_inject_weight.weight"
    assert by_name["model.language_model.layers.1.ple.conv1d.weight"][
        "module_weight_key"
    ] == "layers.1.ple.conv1d.weight"

    table = by_name[
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_127.weight"
    ]
    assert table["kind"] == "provider"
    assert table["component"] == "ple"
    assert table["provider_tensor_kind"] == "embedding_table"
    assert table["provider_shard_index"] == 127
    assert table["format"] == "raw_dtype_passthrough"
    assert "module_path" not in table
    assert "module_weight_key" not in table

    metadata = by_name["model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets"]
    assert metadata["role"] == "ple.provider.ngram_heads_offsets"
    assert metadata["provider_tensor_kind"] == "metadata"
    assert all(not entry["gguf_keys"] for entry in inventory["tensors"])


def test_qwen4_inventory_resolves_every_graph_tensor_to_the_injected_shell() -> None:
    inventory = _inventory(_released_served_headers())
    graph = [entry for entry in inventory["tensors"] if entry["kind"] != "provider"]
    ordinary = [entry for entry in graph if entry["kind"] != "expert"]
    experts = [entry for entry in graph if entry["kind"] == "expert"]

    assert len(graph) == EXPECTED_TEXT_TENSOR_COUNT
    assert all(isinstance(entry.get("module_path"), str) for entry in graph)
    assert all(isinstance(entry.get("module_weight_key"), str) for entry in ordinary)
    assert len({entry["module_weight_key"] for entry in ordinary}) == len(ordinary)
    assert all("module_weight_key" not in entry for entry in experts)
    assert {entry["module_path"] for entry in experts} == {
        f"layers.{layer}.mlp.experts" for layer in range(48)
    }
    router = next(
        entry
        for entry in ordinary
        if entry["source_name"] == "model.language_model.layers.17.mlp.gate.weight"
    )
    assert router["module_weight_key"] == "layers.17.mlp.gate.weight"


def test_qwen4_inventory_unknown_served_tensor_is_blocking() -> None:
    headers = _released_served_headers() + [
        _header(
            "model.language_model.layers.0.unexpected.weight",
            "BF16",
            (2560, 2560),
        )
    ]
    inventory = _inventory(headers)

    assert inventory["status"] == "invalid"
    assert inventory["counts"]["unknown"] == 1
    assert any(
        issue["code"] == "inventory.unknown_tensors" and issue["blocking"]
        for issue in inventory["validation"]
    )


def test_qwen4_inventory_rejects_known_attention_name_on_wrong_layer_kind() -> None:
    headers = [
        _header(
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "BF16",
            (12288, 2560),
        )
    ]
    inventory = _inventory(headers, family="qwen4_exp_text")

    assert inventory["status"] == "invalid"
    assert inventory["counts"]["unknown"] == 1
    assert inventory["family"] == "qwen4_exp"


def test_qwen4_inventory_requires_every_released_served_tensor() -> None:
    headers = _released_served_headers()
    missing = "model.language_model.layers.47.mlp.shared_expert_gate.weight"
    inventory = _inventory([header for header in headers if header.name != missing])

    assert inventory["status"] == "invalid"
    assert any(
        issue["code"] == "qwen4.missing_tensor" and missing in issue["path"]
        for issue in inventory["validation"]
    )


def test_qwen4_inventory_rejects_duplicate_served_tensor() -> None:
    headers = _released_served_headers()
    inventory = _inventory(headers + [headers[0]])

    assert inventory["status"] == "invalid"
    assert any(issue["code"] == "qwen4.duplicate_tensor" for issue in inventory["validation"])


def test_qwen4_inventory_rejects_known_wrong_dtype_and_shape() -> None:
    headers = _released_served_headers()
    target = "model.language_model.layers.3.self_attn.q_proj.weight"
    wrong = _header(target, "F16", (2560, 12288))
    inventory = _inventory([wrong if header.name == target else header for header in headers])

    assert inventory["status"] == "invalid"
    codes = {issue["code"] for issue in inventory["validation"]}
    assert "qwen4.dtype_mismatch" in codes
    assert "qwen4.shape_mismatch" in codes
