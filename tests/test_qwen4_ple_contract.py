"""Pure architecture authority for package-owned Qwen4 PLE rows."""

from __future__ import annotations

from copy import deepcopy

import pytest

from moespresso.runtime.qwen4.ple_contract import (
    Qwen4PLEProviderError,
    derive_qwen4_ple_provider_contract,
)
from moespresso.runtime.verify import verify_package


_TABLE_SIZES = (
    20000003,
    20000023,
    20000033,
    20000047,
    20000059,
    20000063,
    20000069,
    20000077,
    20000081,
    20000093,
    20000107,
    20000147,
    20000153,
    20000159,
    20000161,
    20000171,
)
_TABLE_OFFSETS = (
    0,
    20000003,
    40000026,
    60000059,
    80000106,
    100000165,
    120000228,
    140000297,
    160000374,
    180000455,
    200000548,
    220000655,
    240000802,
    260000955,
    280001114,
    300001275,
)


def _architecture() -> dict:
    layer_types = ["linear_attention"] * 48
    for index in range(3, 48, 4):
        layer_types[index] = "full_attention"
    return {
        "family": "qwen4_exp",
        "text_model_type": "qwen4_exp_text",
        "config": {
            "model_type": "qwen4_exp_text",
            "dtype": "bfloat16",
            "hidden_size": 2560,
            "num_hidden_layers": 48,
            "layer_types": layer_types,
            "ple_layer_ids": [2],
            "ngram_size": 3,
            "heads_per_ngram": 8,
            "ple_embed_dim": 2560,
            "vocab_size": 248320,
            "seed": 1234,
            "ngram_vocab_size_base": 20000000,
            "make_ngram_vocab_size_divisible_by": 128,
            "split_ngram_parts": 128,
        },
    }


def _component() -> dict:
    contract = derive_qwen4_ple_provider_contract(_architecture())
    assert contract is not None
    return {
        "schema": "qwen4_ple_provider_v1",
        "layer_index": contract.layer_index,
        "dtype": contract.dtype,
        "row_width": contract.row_width,
        "row_bytes": contract.row_bytes,
        "logical_rows": contract.logical_rows,
        "padded_rows": contract.padded_rows,
        "rows_per_shard": contract.rows_per_shard,
        "ngram_size": contract.ngram_size,
        "heads_per_ngram": contract.heads_per_ngram,
        "multipliers": list(contract.multipliers),
        "table_sizes": list(contract.table_sizes),
        "table_offsets": list(contract.table_offsets),
        "shards": [
            {
                "index": index,
                "path": f"ple/rows-{index:03d}-of-128.bf16",
                "row_start": index * contract.rows_per_shard,
                "row_count": contract.rows_per_shard,
            }
            for index in range(contract.shard_count)
        ],
    }


def test_released_architecture_derives_exact_ple_contract() -> None:
    contract = derive_qwen4_ple_provider_contract(_architecture())

    assert contract is not None
    assert contract.layer_index == 1
    assert (contract.row_width, contract.row_bytes) == (160, 320)
    assert (contract.logical_rows, contract.padded_rows) == (320001446, 320001536)
    assert (contract.shard_count, contract.rows_per_shard) == (128, 2500012)
    assert contract.multipliers == (
        23703573157769,
        20109073645365,
        8052911324071,
    )
    assert contract.table_sizes == _TABLE_SIZES
    assert contract.table_offsets == _TABLE_OFFSETS


def test_architecture_derivation_refuses_implicit_hash_seed() -> None:
    architecture = _architecture()
    del architecture["config"]["seed"]

    with pytest.raises(Qwen4PLEProviderError, match="seed"):
        derive_qwen4_ple_provider_contract(architecture)


def test_package_verifier_requires_provider_for_declared_ple(tmp_path) -> None:
    issues = verify_package(
        {"architecture": _architecture(), "files": [], "tensors": []},
        tmp_path,
    )

    assert any(issue.code == "runtime.missing_qwen4_ple_provider" for issue in issues)


def test_package_verifier_refuses_self_consistent_wrong_hash_contract(tmp_path) -> None:
    component = deepcopy(_component())
    component["multipliers"][0] += 2
    issues = verify_package(
        {
            "architecture": _architecture(),
            "ple_provider": component,
            "files": [],
            "tensors": [],
        },
        tmp_path,
    )

    mismatch = [
        issue for issue in issues if issue.code == "runtime.qwen4_ple_contract_mismatch"
    ]
    assert mismatch
    assert "multipliers" in mismatch[0].message


def test_package_verifier_binds_ple_shards_to_top_level_identities(tmp_path) -> None:
    issues = verify_package(
        {
            "architecture": _architecture(),
            "ple_provider": _component(),
            "files": [],
            "tensors": [],
        },
        tmp_path,
    )

    undeclared = [
        issue for issue in issues if issue.code == "runtime.undeclared_qwen4_ple_shard"
    ]
    assert len(undeclared) == 128
