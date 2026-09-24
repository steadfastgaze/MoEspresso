"""Manifest and direct-row tests for Qwen4 PLE storage."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path
import threading

import mlx.core as mx
import numpy as np
import pytest

import moespresso.runtime.qwen4.ple_provider as provider_module
from moespresso.runtime.qwen4.ple import (
    Qwen4NGramHasher,
    Qwen4PLELayer,
)
from moespresso.runtime.qwen4.ple_provider import (
    Qwen4PLEDirectRowProvider,
    Qwen4PLEProviderContract,
    Qwen4PLEProviderError,
    Qwen4PLEProviderLayout,
    Qwen4PLEShardRecord,
    parse_qwen4_ple_provider,
)


def _bf16_bytes(values: np.ndarray) -> bytes:
    array = mx.array(values, dtype=mx.bfloat16)
    mx.eval(array)
    return bytes(memoryview(array).cast("B"))


def _write_shards(
    root: Path,
    *,
    shard_count: int,
    rows_per_shard: int,
    row_width: int,
) -> tuple[Qwen4PLEShardRecord, ...]:
    shards = []
    for index in range(shard_count):
        row_start = index * rows_per_shard
        values = np.stack(
            [
                np.arange(row_start, row_start + rows_per_shard, dtype=np.float32),
                np.arange(row_start, row_start + rows_per_shard, dtype=np.float32) + 0.5,
            ],
            axis=1,
        )[:, :row_width]
        path = root / f"ple/shard-{index:03d}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = _bf16_bytes(values)
        path.write_bytes(payload)
        shards.append(
            Qwen4PLEShardRecord(
                index=index,
                path=f"ple/{path.name}",
                file_path=path,
                row_start=row_start,
                row_count=rows_per_shard,
                size_bytes=len(payload),
                sha256=f"{index:064x}",
            )
        )
    return tuple(shards)


def _layout(
    root: Path,
    *,
    shard_count: int = 3,
    rows_per_shard: int = 4,
    logical_rows: int | None = None,
) -> Qwen4PLEProviderLayout:
    padded_rows = shard_count * rows_per_shard
    logical = padded_rows if logical_rows is None else logical_rows
    return Qwen4PLEProviderLayout(
        layer_index=1,
        dtype="BF16",
        row_width=2,
        row_bytes=4,
        logical_rows=logical,
        padded_rows=padded_rows,
        rows_per_shard=rows_per_shard,
        ngram_size=2,
        heads_per_ngram=1,
        multipliers=(3, 5),
        table_sizes=(logical,),
        table_offsets=(0,),
        shards=_write_shards(
            root,
            shard_count=shard_count,
            rows_per_shard=rows_per_shard,
            row_width=2,
        ),
    )


def _manifest(
    root: Path,
    *,
    shard_count: int = 3,
    rows_per_shard: int = 4,
    logical_rows: int = 11,
) -> dict:
    layout = _layout(
        root,
        shard_count=shard_count,
        rows_per_shard=rows_per_shard,
        logical_rows=logical_rows,
    )
    return {
        "files": [
            {
                "path": shard.path,
                "size_bytes": shard.size_bytes,
                "sha256": shard.sha256,
            }
            for shard in layout.shards
        ],
        "ple_provider": {
            "schema": "qwen4_ple_provider_v1",
            "layer_index": 1,
            "dtype": "BF16",
            "row_width": 2,
            "row_bytes": 4,
            "logical_rows": logical_rows,
            "padded_rows": shard_count * rows_per_shard,
            "rows_per_shard": rows_per_shard,
            "ngram_size": 2,
            "heads_per_ngram": 1,
            "multipliers": [3, 5],
            "table_sizes": [logical_rows],
            "table_offsets": [0],
            "shards": [
                {
                    "index": shard.index,
                    "path": shard.path,
                    "row_start": shard.row_start,
                    "row_count": shard.row_count,
                }
                for shard in layout.shards
            ],
        },
    }


def _contract(
    *,
    shard_count: int = 3,
    rows_per_shard: int = 4,
    logical_rows: int = 11,
) -> Qwen4PLEProviderContract:
    return Qwen4PLEProviderContract(
        layer_index=1,
        dtype="BF16",
        row_width=2,
        row_bytes=4,
        logical_rows=logical_rows,
        padded_rows=shard_count * rows_per_shard,
        rows_per_shard=rows_per_shard,
        shard_count=shard_count,
        ngram_size=2,
        heads_per_ngram=1,
        multipliers=(3, 5),
        table_sizes=(logical_rows,),
        table_offsets=(0,),
    )


def _direct_ple_layer(provider: Qwen4PLEDirectRowProvider) -> Qwen4PLELayer:
    layer = Qwen4PLELayer(
        Qwen4NGramHasher(
            eos_token_id=9,
            ngram_size=2,
            heads_per_ngram=1,
            multipliers=np.array([3, 5], dtype=np.int64),
            table_sizes=np.array([12], dtype=np.int64),
            table_offsets=np.array([0], dtype=np.int64),
        ),
        provider,
        row_width=2,
        hidden_size=2,
        branch_count=2,
        conv_kernel_size=2,
        conv_dilation=2,
    )
    layer.key_proj.weight = mx.array(
        [
            [0.2, -0.1],
            [-0.4, 0.2],
            [0.1, 0.5],
            [0.3, -0.3],
        ],
        dtype=mx.float32,
    )
    layer.value_proj.weight = mx.array(
        [[0.5, -0.2], [-0.1, 0.4]],
        dtype=mx.float32,
    )
    layer.norm_key.weight = mx.zeros((4,), dtype=mx.float32)
    layer.norm_query.weight = mx.zeros((4,), dtype=mx.float32)
    layer.norm_conv.weight = mx.zeros((4,), dtype=mx.float32)
    layer.conv1d.weight = mx.array(
        [
            [[0.2], [0.5]],
            [[-0.3], [0.1]],
            [[0.4], [-0.2]],
            [[0.1], [0.3]],
        ],
        dtype=mx.float32,
    )
    return layer


def test_provider_maps_first_second_and_last_of_128_shards(tmp_path: Path) -> None:
    layout = _layout(tmp_path, shard_count=128, rows_per_shard=2)
    provider = Qwen4PLEDirectRowProvider(layout)

    assert provider.file_cache.max_open == 128

    got = provider.lookup(mx.array([0, 2, 254], dtype=mx.int64))
    mx.eval(got)
    provider.close()

    assert np.array_equal(
        np.asarray(got.astype(mx.float32)),
        np.array([[0, 0.5], [2, 2.5], [254, 254]], dtype=np.float32),
    )


def test_lookup_preserves_order_and_duplicates_with_coalesced_reads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    calls = []
    allocations = []
    real_pread = provider_module.pread_view_cached
    real_zeros = provider_module.mx.zeros

    def tracked_pread(view, path, *, file_offset, nbytes, dst_offset=0, cache):
        calls.append((Path(path).name, file_offset, nbytes, dst_offset))
        return real_pread(
            view,
            path,
            file_offset=file_offset,
            nbytes=nbytes,
            dst_offset=dst_offset,
            cache=cache,
        )

    def tracked_zeros(shape, *args, **kwargs):
        allocations.append(tuple(shape))
        return real_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(provider_module, "pread_view_cached", tracked_pread)
    monkeypatch.setattr(provider_module.mx, "zeros", tracked_zeros)
    provider = Qwen4PLEDirectRowProvider(layout)

    got = provider.lookup(mx.array([[5, 1, 2, 1, 4]], dtype=mx.int64))
    mx.eval(got)
    provider.close()

    assert np.array_equal(
        np.asarray(got.astype(mx.float32)),
        np.array([[[5, 5.5], [1, 1.5], [2, 2.5], [1, 1.5], [4, 4.5]]]),
    )
    assert allocations == [(4, 2)]
    assert calls == [
        ("shard-000.bin", 4, 8, 0),
        ("shard-001.bin", 0, 8, 8),
    ]
    assert sum(call[2] for call in calls) == 4 * layout.row_bytes


def test_lookup_preserves_raw_bf16_rows_bit_for_bit(tmp_path: Path) -> None:
    layout = _layout(tmp_path, shard_count=1, rows_per_shard=2)
    patterns = np.array(
        [[0x3F80, 0xC020], [0x0001, 0x7F7F]],
        dtype=np.uint16,
    )
    layout.shards[0].file_path.write_bytes(patterns.tobytes())
    provider = Qwen4PLEDirectRowProvider(layout)

    got = provider.lookup(mx.array([1, 0, 1], dtype=mx.int64))
    mx.eval(got)
    provider.close()

    expected = patterns[[1, 0, 1]].tobytes()
    assert bytes(memoryview(got).cast("B")) == expected


def test_provider_closes_only_its_owned_file_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    owned = Qwen4PLEDirectRowProvider(_layout(tmp_path / "owned"))
    owned_calls = []
    monkeypatch.setattr(owned.file_cache, "close_all", lambda: owned_calls.append(True))
    owned.close()

    borrowed_cache = provider_module.PreadFileCache(max_open=2)
    borrowed_calls = []
    monkeypatch.setattr(
        borrowed_cache,
        "close_all",
        lambda: borrowed_calls.append(True),
    )
    borrowed = Qwen4PLEDirectRowProvider(
        _layout(tmp_path / "borrowed"),
        file_cache=borrowed_cache,
    )
    borrowed.close()

    assert owned_calls == [True]
    assert borrowed_calls == []


@pytest.mark.parametrize(
    ("values", "error", "match"),
    [
        ([-1], ValueError, "nonnegative"),
        ([11], ValueError, "logical PLE row range"),
        ([11.0], TypeError, "contain integers"),
    ],
)
def test_lookup_rejects_negative_padded_and_noninteger_rows(
    tmp_path: Path,
    values,
    error,
    match,
) -> None:
    provider = Qwen4PLEDirectRowProvider(_layout(tmp_path, logical_rows=11))
    with pytest.raises(error, match=match):
        provider.lookup(mx.array(values))
    provider.close()


def test_lookup_fails_closed_on_short_read(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    provider = Qwen4PLEDirectRowProvider(layout)
    layout.shards[1].file_path.write_bytes(b"\x00\x00")

    with pytest.raises(Qwen4PLEProviderError, match="failed to read selected PLE rows"):
        provider.lookup(mx.array([4], dtype=mx.int64))
    provider.close()


def test_parallel_lookup_preserves_duplicates_and_keeps_mlx_on_caller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = Qwen4PLEDirectRowProvider(_layout(tmp_path))
    ids = mx.array([[10, 0, 4, 2, 8, 6, 4]], dtype=mx.int64)
    expected = provider.lookup(ids)
    mx.eval(expected)
    assert provider._read_executor is None
    threads = []
    original_read = provider_module.pread_view_cached
    original_eval = mx.eval
    caller = threading.get_ident()

    def read(*args, **kwargs):
        threads.append(threading.current_thread().name)
        return original_read(*args, **kwargs)

    def evaluate(*args, **kwargs):
        assert threading.get_ident() == caller
        return original_eval(*args, **kwargs)

    monkeypatch.setattr(provider_module, "_PARALLEL_READ_MIN_RUNS", 1)
    monkeypatch.setattr(provider_module, "pread_view_cached", read)
    monkeypatch.setattr(mx, "eval", evaluate)
    try:
        actual = provider.lookup(ids)
        mx.eval(actual)
        assert np.array_equal(
            np.asarray(actual.view(mx.uint16)), np.asarray(expected.view(mx.uint16)),
        )
        assert threads and all(name.startswith("moespresso-qwen4-ple-read") for name in threads)
    finally:
        provider.close()


def test_parallel_read_failure_waits_for_other_active_writers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = Qwen4PLEDirectRowProvider(_layout(tmp_path))
    runs = provider._read_plan(np.array([0, 2, 4, 6], dtype=np.int64))
    view = memoryview(bytearray(4 * provider.layout.row_bytes))
    entered = threading.Event()
    failed = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def read(*args, dst_offset, **kwargs):
        if dst_offset == 0:
            assert entered.wait(3)
            failed.set()
            raise OSError("injected PLE read failure")
        if dst_offset == provider.layout.row_bytes:
            entered.set()
            assert release.wait(3)
            finished.set()
        return kwargs["nbytes"]

    monkeypatch.setattr(provider_module, "_PARALLEL_READ_MIN_RUNS", 1)
    monkeypatch.setattr(provider_module, "pread_view_cached", read)
    try:
        with ThreadPoolExecutor(max_workers=1) as caller:
            future = caller.submit(provider._read_runs, view, runs)
            try:
                assert failed.wait(3)
                assert not future.done()
            finally:
                release.set()
            with pytest.raises(OSError, match="injected PLE read failure"):
                future.result(timeout=3)
            assert finished.is_set()
    finally:
        release.set()
        provider.close()


def test_close_drains_parallel_reads_before_closing_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = Qwen4PLEDirectRowProvider(_layout(tmp_path))
    runs = provider._read_plan(np.array([0, 2, 4, 6], dtype=np.int64))
    view = memoryview(bytearray(4 * provider.layout.row_bytes))
    entered = threading.Event()
    closing = threading.Event()
    release = threading.Event()
    file_close = threading.Event()
    original_close = provider.file_cache.close_all

    def read(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        assert not file_close.is_set()
        return kwargs["nbytes"]

    def close_files():
        file_close.set()
        original_close()

    def close_provider():
        closing.set()
        provider.close()

    monkeypatch.setattr(provider_module, "_PARALLEL_READ_MIN_RUNS", 1)
    monkeypatch.setattr(provider_module, "pread_view_cached", read)
    monkeypatch.setattr(provider.file_cache, "close_all", close_files)
    try:
        with ThreadPoolExecutor(max_workers=2) as callers:
            reading = callers.submit(provider._read_runs, view, runs)
            assert entered.wait(3)
            shutdown = callers.submit(close_provider)
            try:
                assert closing.wait(3)
                assert not file_close.is_set()
            finally:
                release.set()
            shutdown.result(timeout=3)
            try:
                reading.result(timeout=3)
            except RuntimeError:
                # Shutdown may reject or cancel a not-yet-started read batch.
                pass
            assert file_close.is_set()
    finally:
        release.set()
        provider.close()


def test_parser_binds_immutable_shards_to_top_level_identities(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)

    got = parse_qwen4_ple_provider(manifest, tmp_path, expected=_contract())

    assert got.logical_rows == 11
    assert [shard.index for shard in got.shards] == [0, 1, 2]
    assert got.shards[2].sha256 == manifest["files"][2]["sha256"]
    with pytest.raises(FrozenInstanceError):
        got.shards[0].row_start = 9


def test_parser_binds_all_128_provider_shards_to_the_model_contract(tmp_path: Path) -> None:
    manifest = _manifest(
        tmp_path,
        shard_count=128,
        rows_per_shard=2,
        logical_rows=255,
    )

    got = parse_qwen4_ple_provider(
        manifest,
        tmp_path,
        expected=_contract(
            shard_count=128,
            rows_per_shard=2,
            logical_rows=255,
        ),
    )

    assert len(got.shards) == 128
    assert got.shards[-1].row_start == 254


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("layer_index", 2),
        ("multipliers", [7, 5]),
    ],
)
def test_parser_rejects_self_consistent_component_geometry_not_owned_by_model(
    tmp_path: Path,
    field: str,
    value,
) -> None:
    manifest = _manifest(tmp_path)
    manifest["ple_provider"][field] = value

    with pytest.raises(Qwen4PLEProviderError, match=f"model contract: {field}"):
        parse_qwen4_ple_provider(manifest, tmp_path, expected=_contract())


def test_parser_rejects_missing_shard_file(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    (tmp_path / manifest["files"][1]["path"]).unlink()

    with pytest.raises(Qwen4PLEProviderError, match="is missing"):
        parse_qwen4_ple_provider(manifest, tmp_path, expected=_contract())


@pytest.mark.parametrize("path", ["../escape.bin", "/tmp/escape.bin", "ple\\escape.bin"])
def test_parser_rejects_path_escape(tmp_path: Path, path: str) -> None:
    manifest = _manifest(tmp_path)
    manifest["files"][0]["path"] = path
    manifest["ple_provider"]["shards"][0]["path"] = path

    with pytest.raises(Qwen4PLEProviderError, match="package root|canonical"):
        parse_qwen4_ple_provider(manifest, tmp_path, expected=_contract())


def test_parser_rejects_symlink_escape(tmp_path: Path) -> None:
    package = tmp_path / "package"
    manifest = _manifest(package)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"\x00" * 16)
    link = package / "ple/escape-link.bin"
    link.symlink_to(outside)
    manifest["files"][0]["path"] = "ple/escape-link.bin"
    manifest["ple_provider"]["shards"][0]["path"] = "ple/escape-link.bin"

    with pytest.raises(Qwen4PLEProviderError, match="package root"):
        parse_qwen4_ple_provider(manifest, package, expected=_contract())


def test_parser_rejects_shard_order(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    manifest["ple_provider"]["shards"][0], manifest["ple_provider"]["shards"][1] = (
        manifest["ple_provider"]["shards"][1],
        manifest["ple_provider"]["shards"][0],
    )

    with pytest.raises(Qwen4PLEProviderError, match="ordered by contiguous index"):
        parse_qwen4_ple_provider(manifest, tmp_path, expected=_contract())


def test_parser_rejects_identity_size_mismatch(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    manifest["files"][0]["size_bytes"] -= 1

    with pytest.raises(Qwen4PLEProviderError, match="identity size"):
        parse_qwen4_ple_provider(manifest, tmp_path, expected=_contract())


def test_parser_rejects_physical_file_size_mismatch(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    shard = tmp_path / manifest["files"][0]["path"]
    shard.write_bytes(shard.read_bytes()[:-1])

    with pytest.raises(Qwen4PLEProviderError, match="has size"):
        parse_qwen4_ple_provider(manifest, tmp_path, expected=_contract())


def test_parser_rejects_unbound_or_duplicate_shard_paths(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    manifest["ple_provider"]["shards"][1]["path"] = "ple/not-declared.bin"
    with pytest.raises(Qwen4PLEProviderError, match="no top-level file identity"):
        parse_qwen4_ple_provider(manifest, tmp_path, expected=_contract())

    manifest = _manifest(tmp_path / "duplicate")
    manifest["ple_provider"]["shards"][1]["path"] = manifest["ple_provider"]["shards"][0]["path"]
    with pytest.raises(Qwen4PLEProviderError, match="duplicate shard path"):
        parse_qwen4_ple_provider(
            manifest,
            tmp_path / "duplicate",
            expected=_contract(),
        )


def test_parser_rejects_duplicate_top_level_identity(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    manifest["files"].append(dict(manifest["files"][0]))

    with pytest.raises(Qwen4PLEProviderError, match="duplicate file identity"):
        parse_qwen4_ple_provider(manifest, tmp_path, expected=_contract())
