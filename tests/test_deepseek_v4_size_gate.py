from __future__ import annotations

import json

import pytest

from moespresso.package.convert import (
    DEEPSEEK_V4_ABSOLUTE_PACKAGE_SIZE_GB,
    _deepseek_v4_package_size_contract,
    _source_safetensors_total_size,
)


def _write_index(path, total_size):
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": {}}),
        encoding="utf-8",
    )


def _manifest(*sizes):
    return {"files": [{"path": f"model-{i}.safetensors", "size_bytes": s}
                      for i, s in enumerate(sizes)]}


def test_source_safetensors_total_size_reads_index_metadata(tmp_path):
    _write_index(tmp_path, 1234)

    assert _source_safetensors_total_size(tmp_path) == 1234


def test_deepseek_v4_size_gate_records_passing_contract(tmp_path):
    _write_index(tmp_path, 1000)

    contract = _deepseek_v4_package_size_contract(
        "deepseek_v4_flash",
        tmp_path,
        _manifest(400, 500),
    )

    assert contract == {
        "absolute_package_size_gb": 85.0,
        "family": "deepseek_v4_flash",
        "package_le_absolute_ceiling": True,
        "source_size_bytes": 1000,
        "package_size_bytes": 900,
        "package_le_source": True,
        "practical_target_size_gb": 75.0,
    }


def test_deepseek_v4_size_gate_raises_when_package_exceeds_source(tmp_path):
    _write_index(tmp_path, 1000)

    with pytest.raises(RuntimeError, match="package size gate FAILED"):
        _deepseek_v4_package_size_contract(
            "deepseek_v4_flash",
            tmp_path,
            _manifest(700, 400),
        )


def test_deepseek_v4_size_gate_raises_when_package_exceeds_absolute_ceiling(tmp_path):
    ceiling = int(DEEPSEEK_V4_ABSOLUTE_PACKAGE_SIZE_GB * (1024 ** 3))
    _write_index(tmp_path, ceiling * 2)

    with pytest.raises(RuntimeError, match="absolute 85.0 GiB ceiling"):
        _deepseek_v4_package_size_contract(
            "deepseek_v4_flash",
            tmp_path,
            _manifest(ceiling + 1),
        )


def test_deepseek_v4_size_gate_requires_source_index(tmp_path):
    with pytest.raises(RuntimeError, match="metadata.total_size"):
        _deepseek_v4_package_size_contract(
            "deepseek_v4_flash",
            tmp_path,
            _manifest(1),
        )


def test_package_size_gate_is_deepseek_only(tmp_path):
    assert _deepseek_v4_package_size_contract("qwen3_5_moe", tmp_path, _manifest(999)) is None
