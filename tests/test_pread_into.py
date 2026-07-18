"""Direct pread-into-buffer primitive."""

from __future__ import annotations

import json
import os
import struct

import pytest

from moespresso.runtime.expert_index import build_expert_index
from moespresso.runtime.pread_into import (
    PreadFileCache,
    PreadIntoShortRead,
    pread_into,
    pread_into_cached,
)


def test_pread_into_byte_array(tmp_path):
    mx = pytest.importorskip("mlx.core")
    payload = bytes(range(64))
    source = tmp_path / "source.bin"
    source.write_bytes(payload)
    dst = mx.zeros((64,), dtype=mx.uint8)
    mx.eval(dst)

    n = pread_into(dst, source, file_offset=4, nbytes=16, dst_offset=32)
    mx.eval(dst)

    assert n == 16
    assert bytes(memoryview(dst)[32:48]) == payload[4:20]
    assert bytes(memoryview(dst)[:32]) == bytes(32)


def test_pread_into_typed_mlx_array_uses_byte_offsets(tmp_path):
    mx = pytest.importorskip("mlx.core")
    payload = bytes(range(64))
    source = tmp_path / "source.bin"
    source.write_bytes(payload)
    dst = mx.zeros((4,), dtype=mx.uint32)
    mx.eval(dst)

    pread_into(dst, source, file_offset=8, nbytes=8, dst_offset=4)
    mx.eval(dst)

    assert bytes(memoryview(dst).cast("B")[4:12]) == payload[8:16]


def _write_safetensors(path, tensors, metadata=None):
    header, blob, off = {}, bytearray(), 0
    if metadata:
        header["__metadata__"] = dict(metadata)
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [off, off + len(raw)],
        }
        blob += raw
        off += len(raw)
    hjson = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        f.write(blob)


def test_pread_into_safetensors_expert_range_lands_in_pool_slot(tmp_path):
    mx = pytest.importorskip("mlx.core")
    import numpy as np

    from moespresso.package.bundle import (
        assemble_layer_bundle,
        encode_bundle_metadata,
    )

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    base = "language_model.model.layers.0.mlp.switch_mlp"
    packed = bytes(range(64))  # 4 experts * 2 rows * 2 packed cols * sizeof(U32)
    comps = {
        ("gate_proj", "packed"): np.frombuffer(packed, np.uint32).reshape(4, 2, 2),
        ("gate_proj", "norms"): np.zeros((4, 2), np.float16),
        ("up_proj", "packed"): np.zeros((4, 2, 2), np.uint32),
        ("up_proj", "norms"): np.zeros((4, 2), np.float16),
        ("down_proj", "packed"): np.zeros((4, 2, 2), np.uint32),
        ("down_proj", "norms"): np.zeros((4, 2), np.float16),
    }
    bundle, geo = assemble_layer_bundle(
        comps, {p: 2 for p in ("gate_proj", "up_proj", "down_proj")})
    _write_safetensors(pkg / "model-00001-of-00001.safetensors", {
        f"{base}.experts.tq_bundle": ("U8", bundle.shape, bundle.tobytes()),
    }, metadata={"expert_bundles": encode_bundle_metadata({0: geo})})
    index = build_expert_index(pkg)
    expert = index.locate(
        layer=0, expert=2, projection="gate_proj", component="packed")
    pool = mx.zeros((4, 2, 2), dtype=mx.uint32)
    mx.eval(pool)

    pread_into(
        pool,
        pkg / expert.shard,
        file_offset=expert.offset,
        nbytes=expert.nbytes,
        dst_offset=expert.nbytes,
    )
    mx.eval(pool)

    pool_bytes = memoryview(pool).cast("B")
    assert bytes(pool_bytes[expert.nbytes:2 * expert.nbytes]) == packed[32:48]
    assert bytes(pool_bytes[:expert.nbytes]) == bytes(expert.nbytes)


def test_pread_into_loops_over_short_syscall(monkeypatch, tmp_path):
    mx = pytest.importorskip("mlx.core")
    payload = b"abcdefghijklmnop"
    source = tmp_path / "source.bin"
    source.write_bytes(payload)
    real_preadv = os.preadv
    calls = []

    def short_preadv(fd, buffers, offset):
        calls.append(offset)
        view = buffers[0]
        return real_preadv(fd, [view[:3]], offset)

    monkeypatch.setattr(os, "preadv", short_preadv)
    dst = mx.zeros((16,), dtype=mx.uint8)
    mx.eval(dst)

    assert pread_into(dst, source, file_offset=0, nbytes=9) == 9
    assert calls == [0, 3, 6]
    assert bytes(memoryview(dst)[:9]) == payload[:9]


def test_pread_into_cached_reuses_and_evicts_fds(monkeypatch, tmp_path):
    mx = pytest.importorskip("mlx.core")
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"abcdefgh")
    second.write_bytes(b"ABCDEFGH")
    dst = mx.zeros((8,), dtype=mx.uint8)
    mx.eval(dst)

    real_open = os.open
    real_close = os.close
    opened = []
    closed = []

    def tracked_open(path, flags, mode=0o777):
        opened.append(os.fspath(path))
        return real_open(path, flags, mode)

    def tracked_close(fd):
        closed.append(fd)
        real_close(fd)

    monkeypatch.setattr(os, "open", tracked_open)
    monkeypatch.setattr(os, "close", tracked_close)
    cache = PreadFileCache(max_open=1)

    pread_into_cached(dst, first, file_offset=0, nbytes=1, cache=cache)
    pread_into_cached(dst, first, file_offset=1, nbytes=1, dst_offset=1, cache=cache)
    pread_into_cached(dst, second, file_offset=0, nbytes=1, dst_offset=2, cache=cache)
    cache.close_all()

    assert opened == [os.fspath(first), os.fspath(second)]
    assert len(closed) == 2


def test_pread_file_cache_does_not_evict_leased_fd(monkeypatch, tmp_path):
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"abcdefgh")
    second.write_bytes(b"ABCDEFGH")

    real_close = os.close
    closed = []

    def tracked_close(fd):
        closed.append(fd)
        real_close(fd)

    monkeypatch.setattr(os, "close", tracked_close)
    cache = PreadFileCache(max_open=1)

    with cache.acquire_fd(first) as first_fd:
        second_fd = cache.fd(second)
        assert first_fd not in closed
        assert second_fd not in closed
    cache.close_all()

    assert first_fd in closed
    assert second_fd in closed


def test_pread_into_rejects_out_of_bounds_destination(tmp_path):
    mx = pytest.importorskip("mlx.core")
    source = tmp_path / "source.bin"
    source.write_bytes(b"abc")
    dst = mx.zeros((2,), dtype=mx.uint8)
    mx.eval(dst)

    with pytest.raises(ValueError, match="exceeds destination size"):
        pread_into(dst, source, file_offset=0, nbytes=3)


def test_pread_into_raises_on_short_file(tmp_path):
    mx = pytest.importorskip("mlx.core")
    source = tmp_path / "source.bin"
    source.write_bytes(b"abc")
    dst = mx.zeros((8,), dtype=mx.uint8)
    mx.eval(dst)

    with pytest.raises(PreadIntoShortRead, match="got 3 of 8"):
        pread_into(dst, source, file_offset=0, nbytes=8)
