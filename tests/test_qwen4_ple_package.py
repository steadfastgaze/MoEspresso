"""Package-owned Qwen4 PLE row writer tests."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from conftest import write_safetensors_raw
from moespresso.package.qwen4.ple_provider import (
    Qwen4PLEWriteError,
    inspect_qwen4_ple_reuse,
    inspect_qwen4_ple_source,
    write_qwen4_ple_provider,
)
from moespresso.runtime.qwen4.ple_provider import (
    Qwen4PLEDirectRowProvider,
    Qwen4PLEProviderContract,
    parse_qwen4_ple_provider,
)


def _contract() -> Qwen4PLEProviderContract:
    return Qwen4PLEProviderContract(
        layer_index=1,
        dtype="BF16",
        row_width=2,
        row_bytes=4,
        logical_rows=5,
        padded_rows=6,
        rows_per_shard=3,
        shard_count=2,
        ngram_size=2,
        heads_per_ngram=1,
        multipliers=(3, 5),
        table_sizes=(5,),
        table_offsets=(0,),
    )


def _bf16_payload(values: np.ndarray) -> bytes:
    array = mx.array(values, dtype=mx.bfloat16)
    mx.eval(array)
    return bytes(memoryview(array).cast("B"))


def _source(root: Path, *, second_dtype: str = "BF16") -> tuple[bytes, bytes]:
    root.mkdir(parents=True)
    first = _bf16_payload(np.array([[0, 0.5], [1, 1.5], [2, 2.5]], dtype=np.float32))
    second = _bf16_payload(np.array([[3, 3.5], [4, 4.5], [5, 5.5]], dtype=np.float32))
    names = []
    for index, (payload, dtype) in enumerate(((first, "BF16"), (second, second_dtype))):
        name = (
            "model.language_model.layers.1.ple.ple_embedding."
            f"ngram_embedding.shard_{index}.weight"
        )
        shard = f"model-{index + 1:05d}-of-00002.safetensors"
        write_safetensors_raw(
            root / shard,
            {name: (dtype, (3, 2), payload)},
        )
        names.append((name, shard))
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict(names)})
    )
    return first, second


def _hub_snapshot_source(root: Path) -> Path:
    snapshot = root / "models--Qwen--Test/snapshots/revision"
    _source(snapshot)
    blobs = root / "models--Qwen--Test/blobs"
    blobs.mkdir(parents=True)
    for index in range(2):
        shard = snapshot / f"model-{index + 1:05d}-of-00002.safetensors"
        blob = blobs / f"blob-{index}"
        shard.replace(blob)
        shard.symlink_to(blob)
    return snapshot


def test_writer_copies_exact_payloads_and_builds_parseable_component(tmp_path: Path):
    source = tmp_path / "source"
    package = tmp_path / "package"
    first, second = _source(source)

    component, files = write_qwen4_ple_provider(
        source,
        package,
        expected=_contract(),
    )
    manifest = {"files": files, "ple_provider": component}
    layout = parse_qwen4_ple_provider(manifest, package, expected=_contract())
    provider = Qwen4PLEDirectRowProvider(layout)
    got = provider.lookup(mx.array([4, 0, 3, 4], dtype=mx.int64))
    mx.eval(got)
    provider.close()

    assert (package / files[0]["path"]).read_bytes() == first
    assert (package / files[1]["path"]).read_bytes() == second
    assert [record["row_start"] for record in component["shards"]] == [0, 3]
    assert np.array_equal(
        np.asarray(got.astype(mx.float32)),
        np.array([[4, 4.5], [0, 0.5], [3, 3.5], [4, 4.5]], dtype=np.float32),
    )


def _write_reuse_package(source: Path, package: Path) -> tuple[dict, list[dict]]:
    component, files = write_qwen4_ple_provider(
        source,
        package,
        expected=_contract(),
    )
    (package / "package_manifest.json").write_text(
        json.dumps(
            {
                "status": "valid",
                "files": files,
                "ple_provider": component,
            }
        )
    )
    return component, files


def test_writer_hardlinks_manifest_verified_reusable_payloads(tmp_path: Path):
    source = tmp_path / "source"
    donor = tmp_path / "donor"
    target = tmp_path / "target"
    _source(source)
    component, files = _write_reuse_package(source, donor)

    reuse = inspect_qwen4_ple_reuse(
        donor,
        expected=_contract(),
        target_dir=target,
    )
    got_component, got_files = write_qwen4_ple_provider(
        source,
        target,
        expected=_contract(),
        reuse_from=donor,
    )

    assert got_component == component
    assert got_files == files
    assert len(reuse) == 2
    for record in files:
        assert (donor / record["path"]).stat().st_ino == (
            target / record["path"]
        ).stat().st_ino


def test_writer_rejects_reusable_payload_digest_drift(tmp_path: Path):
    source = tmp_path / "source"
    donor = tmp_path / "donor"
    target = tmp_path / "target"
    _source(source)
    _component, files = _write_reuse_package(source, donor)
    drifted = donor / files[0]["path"]
    with open(drifted, "r+b") as payload:
        payload.write(b"\xff")

    with pytest.raises(Qwen4PLEWriteError, match="digest differs"):
        write_qwen4_ple_provider(
            source,
            target,
            expected=_contract(),
            reuse_from=donor,
        )

    assert not (target / files[0]["path"]).exists()


def test_source_inspection_preflights_every_table_without_payload_reads(tmp_path: Path):
    source = tmp_path / "source"
    _source(source)

    tables = inspect_qwen4_ple_source(source, expected=_contract())

    assert [table.index for table in tables] == [0, 1]
    assert [table.header.shape for table in tables] == [(3, 2), (3, 2)]
    assert [table.header.end - table.header.begin for table in tables] == [12, 12]


def test_source_inspection_accepts_canonical_hub_blob_symlinks(tmp_path: Path):
    source = _hub_snapshot_source(tmp_path)

    tables = inspect_qwen4_ple_source(source, expected=_contract())

    assert len(tables) == 2
    assert all("/blobs/" in str(table.source_path) for table in tables)


def test_writer_rejects_source_geometry_not_owned_by_contract(tmp_path: Path):
    source = tmp_path / "source"
    _source(source, second_dtype="F16")

    with pytest.raises(Qwen4PLEWriteError, match="expected BF16"):
        write_qwen4_ple_provider(
            source,
            tmp_path / "package",
            expected=_contract(),
        )
    assert not (tmp_path / "package/ple").exists()


def test_writer_rejects_source_index_path_escape(tmp_path: Path):
    source = tmp_path / "source"
    _source(source)
    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    first = next(iter(index["weight_map"]))
    index["weight_map"][first] = "../outside.safetensors"
    index_path.write_text(json.dumps(index))

    with pytest.raises(Qwen4PLEWriteError, match="not canonical"):
        write_qwen4_ple_provider(
            source,
            tmp_path / "package",
            expected=_contract(),
        )


def test_writer_rejects_output_directory_symlink_escape(tmp_path: Path):
    source = tmp_path / "source"
    package = tmp_path / "package"
    outside = tmp_path / "outside"
    _source(source)
    package.mkdir()
    outside.mkdir()
    (package / "ple").symlink_to(outside, target_is_directory=True)

    with pytest.raises(Qwen4PLEWriteError, match="output directory escapes"):
        write_qwen4_ple_provider(
            source,
            package,
            expected=_contract(),
        )
