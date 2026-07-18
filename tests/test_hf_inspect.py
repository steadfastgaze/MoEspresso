"""Remote Hugging Face model inspector."""

from __future__ import annotations

import struct
from unittest.mock import MagicMock, patch

import pytest

from moespresso.probe.gguf_parse import GGUF_MAGIC


def _gguf_string(value: str) -> bytes:
    data = value.encode()
    return struct.pack("<Q", len(data)) + data


def _tiny_gguf_bytes() -> bytes:
    header = struct.pack("<IIQQ", GGUF_MAGIC, 3, 1, 2)
    kvs = bytearray()
    kvs += _gguf_string("general.architecture")
    kvs += struct.pack("<I", 8)
    kvs += _gguf_string("llama")
    kvs += _gguf_string("general.name")
    kvs += struct.pack("<I", 8)
    kvs += _gguf_string("Tiny")

    tensor = bytearray()
    tensor += _gguf_string("token_embd.weight")
    tensor += struct.pack("<I", 2)
    tensor += struct.pack("<Q", 8)
    tensor += struct.pack("<Q", 16)
    tensor += struct.pack("<I", 0)
    tensor += struct.pack("<Q", 0)
    return header + bytes(kvs) + bytes(tensor)


def test_normalizes_blob_url_to_resolve_url():
    from moespresso.inventory.hf_inspect import _normalize_url

    assert (
        _normalize_url("https://huggingface.co/org/repo/blob/main/model.gguf")
        == "https://huggingface.co/org/repo/resolve/main/model.gguf"
    )


def test_rejects_non_huggingface_url():
    from moespresso.inventory.hf_inspect import inspect_url

    with pytest.raises(ValueError, match="Hugging Face"):
        inspect_url("https://example.com/model.gguf")


def test_extracts_repo_id_from_common_huggingface_urls():
    from moespresso.inventory.hf_inspect import _parse_repo_id

    assert _parse_repo_id("https://huggingface.co/unsloth/Qwen3.5-4B/tree/main") == (
        "unsloth/Qwen3.5-4B"
    )
    assert _parse_repo_id("https://huggingface.co/org/repo/blob/main/model.gguf") == (
        "org/repo"
    )
    assert _parse_repo_id("https://huggingface.co/org/repo/") == "org/repo"
    with pytest.raises(ValueError, match="repo_id"):
        _parse_repo_id("https://huggingface.co/org")


def test_compresses_tensor_layer_ranges():
    from moespresso.inventory.hf_inspect import _compress_tensors

    tensors = [
        (f"model.layers.{layer}.mlp.gate_proj.weight", [32, 16], "BF16")
        for layer in range(4)
    ]
    assert _compress_tensors(tensors) == [
        "    model.layers.[0-3].mlp.gate_proj.weight: [32, 16]  BF16"
    ]


def test_inspects_safetensors_repo_with_mocked_headers(capsys):
    from moespresso.inventory.hf_inspect import inspect_url

    config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "hidden_size": 1024,
        "tie_word_embeddings": True,
    }
    index = {
        "metadata": {"total_size": 4096},
        "weight_map": {
            "model.layers.0.weight": "model-00001-of-00002.safetensors",
            "model.layers.1.weight": "model-00002-of-00002.safetensors",
        },
    }
    headers = {
        "model-00001-of-00002.safetensors": {
            "model.layers.0.weight": {
                "dtype": "BF16",
                "shape": [8, 8],
                "data_offsets": [0, 128],
            }
        },
        "model-00002-of-00002.safetensors": {
            "model.layers.1.weight": {
                "dtype": "F32",
                "shape": [8],
                "data_offsets": [0, 32],
            }
        },
    }

    with (
        patch("moespresso.inventory.hf_inspect._fetch_json") as fetch_json,
        patch("moespresso.inventory.hf_inspect._fetch_safetensors_header") as fetch_header,
    ):
        fetch_json.side_effect = lambda url, timeout=30: (
            config if url.endswith("config.json") else index
        )
        fetch_header.side_effect = lambda base, shard: (shard, headers[shard])
        inspect_url("https://huggingface.co/unsloth/Qwen3.5-4B/tree/main")

    out = capsys.readouterr().out
    assert "Safetensors" in out
    assert "unsloth/Qwen3.5-4B" in out
    assert "2 shards" in out
    assert "2 tensors" in out
    assert "model_type: qwen3_5" in out
    assert "BF16: 1 tensors" in out
    assert "F32: 1 tensors" in out
    assert "model.layers.[0-1].weight" not in out


def test_inspects_single_shard_safetensors_repo_without_index(capsys):
    from moespresso.inventory.hf_inspect import inspect_url

    config = {"model_type": "tiny", "hidden_size": 32}
    header = {
        "embed.weight": {"dtype": "F16", "shape": [128, 32], "data_offsets": [0, 8192]}
    }

    with (
        patch("moespresso.inventory.hf_inspect._fetch_json") as fetch_json,
        patch("moespresso.inventory.hf_inspect._fetch_safetensors_header") as fetch_header,
    ):
        fetch_json.side_effect = lambda url, timeout=30: (
            config if url.endswith("config.json") else None
        )
        fetch_header.return_value = ("model.safetensors", header)
        inspect_url("https://huggingface.co/org/tiny")

    out = capsys.readouterr().out
    assert "1 shard" in out
    assert "embed.weight" in out
    assert "8,192 bytes" in out


def test_inspects_gguf_url_with_range_requests(capsys):
    from moespresso.inventory.hf_inspect import inspect_url

    gguf_data = _tiny_gguf_bytes()
    requested_urls: list[str] = []

    def mock_urlopen(req, timeout=None):
        requested_urls.append(req.full_url)
        range_header = req.get_header("Range")
        assert range_header is not None
        start, end = range_header.replace("bytes=", "").split("-")
        chunk = gguf_data[int(start) : int(end) + 1]
        resp = MagicMock()
        resp.status = 206
        resp.read.return_value = chunk
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    with patch("urllib.request.urlopen", mock_urlopen):
        inspect_url("https://huggingface.co/org/repo/blob/main/model.gguf")

    out = capsys.readouterr().out
    assert requested_urls == ["https://huggingface.co/org/repo/resolve/main/model.gguf"]
    assert "GGUF v3" in out
    assert "llama" in out
    assert "token_embd.weight" in out


def test_reads_remote_gguf_metadata_with_range_requests():
    from moespresso.inventory.hf_inspect import read_remote_gguf_metadata

    gguf_data = _tiny_gguf_bytes()

    def mock_urlopen(req, timeout=None):
        range_header = req.get_header("Range")
        assert range_header is not None
        start, end = range_header.replace("bytes=", "").split("-")
        chunk = gguf_data[int(start) : int(end) + 1]
        resp = MagicMock()
        resp.status = 206
        resp.read.return_value = chunk
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    with patch("urllib.request.urlopen", mock_urlopen):
        metadata = read_remote_gguf_metadata(
            "https://huggingface.co/org/repo/blob/main/model.gguf")

    assert metadata.header.tensor_count == 1
    assert metadata.tensor_infos[0].name == "token_embd.weight"
    assert metadata.tensor_infos[0].dimensions == [8, 16]
    assert metadata.tensor_infos[0].type_id == 0


def test_gguf_inspection_fails_when_server_does_not_support_ranges():
    from moespresso.inventory.hf_inspect import inspect_url

    def mock_urlopen(req, timeout=None):
        resp = MagicMock()
        resp.status = 200
        resp.read.return_value = b""
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    with patch("urllib.request.urlopen", mock_urlopen):
        with pytest.raises(ValueError, match="Range"):
            inspect_url("https://huggingface.co/org/repo/resolve/main/model.gguf")


def test_pyproject_exposes_old_and_moespresso_cli_names():
    text = __import__("pathlib").Path("pyproject.toml").read_text()
    assert 'hf-model-inspect = "moespresso.inventory.hf_inspect:main"' in text
    assert 'moespresso-hf-inspect = "moespresso.inventory.hf_inspect:main"' in text
