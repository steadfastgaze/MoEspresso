"""Calibration providers: GGUF and legacy imatrix readers on synthetic files.

Pure (no mlx/jang): hand-write a tiny GGUF imatrix (header + <base>.in_sum2 /
<base>.counts tensor pairs), then confirm the provider reads the per-channel
vectors h = in_sum2/count correctly and emits a faithful calibration identity.
"""

from __future__ import annotations

import hashlib
import struct

import numpy as np
import pytest

from moespresso.probe.calibration import (
    calibration_identity,
    imatrix_expert_counts,
    imatrix_calibration,
    read_imatrix_vectors,
)

_GGUF_MAGIC = 0x46554747
_F32 = 0  # GGUF tensor type id for F32


def _gguf_string(s: bytes) -> bytes:
    return struct.pack("<Q", len(s)) + s


def _write_imatrix(path, pairs):
    """Write a minimal GGUF v3 imatrix file.

    `pairs`: list of (base_name, in_sum2 float32 array, count float). Each becomes
    two tensors `<base>.in_sum2` (1-D, n) and `<base>.counts` (1-D, 1), F32, with
    data laid out 32-byte-aligned after the header (as the reader expects).
    """
    tensors = []  # (name, np.float32 array)
    for base, in_sum2, count in pairs:
        tensors.append((f"{base}.in_sum2", np.asarray(in_sum2, np.float32)))
        tensors.append((f"{base}.counts", np.asarray([count], np.float32)))

    # header: magic, version=3, tensor_count, kv_count=0
    header = struct.pack("<IIQQ", _GGUF_MAGIC, 3, len(tensors), 0)

    # tensor infos: name, n_dims=1, dims..., type_id, offset (relative to data start)
    infos = bytearray()
    offset = 0
    for name, arr in tensors:
        infos += _gguf_string(name.encode())
        infos += struct.pack("<I", 1)              # n_dimensions
        infos += struct.pack("<Q", arr.shape[0])   # dims[0]
        infos += struct.pack("<I", _F32)           # type id
        infos += struct.pack("<Q", offset)         # data offset
        offset += arr.nbytes

    head = header + bytes(infos)
    # data section is 32-byte aligned after the header+infos
    pad = (-len(head)) % 32
    blob = bytearray()
    for _name, arr in tensors:
        blob += arr.tobytes()

    with open(path, "wb") as f:
        f.write(head)
        f.write(b"\x00" * pad)
        f.write(bytes(blob))


def _write_legacy_imatrix(path, entries, *, chunks=0, dataset=""):
    with open(path, "wb") as f:
        f.write(struct.pack("<i", len(entries)))
        for name, ncall, values in entries:
            data = np.asarray(values, np.float32)
            encoded = name.encode()
            f.write(struct.pack("<i", len(encoded)))
            f.write(encoded)
            f.write(struct.pack("<i", ncall))
            f.write(struct.pack("<i", data.size))
            f.write(data.tobytes())
        f.write(struct.pack("<i", chunks))
        encoded_dataset = dataset.encode()
        f.write(struct.pack("<i", len(encoded_dataset)))
        f.write(encoded_dataset)


def test_reads_per_channel_vectors(tmp_path):
    p = tmp_path / "im.gguf"
    sum2 = np.array([4.0, 9.0, 16.0], np.float32)
    _write_imatrix(p, [("blk.0.attn_q.weight", sum2, 2.0)])
    vecs = read_imatrix_vectors(p)
    assert set(vecs) == {"blk.0.attn_q.weight"}
    # h = in_sum2 / count
    np.testing.assert_allclose(vecs["blk.0.attn_q.weight"], sum2 / 2.0)


def test_zero_count_yields_zero_vector(tmp_path):
    p = tmp_path / "im.gguf"
    _write_imatrix(p, [("blk.0.ffn_down.weight", np.array([1.0, 2.0], np.float32), 0.0)])
    vecs = read_imatrix_vectors(p)
    np.testing.assert_array_equal(vecs["blk.0.ffn_down.weight"], np.zeros(2, np.float32))


def test_multiple_keys(tmp_path):
    p = tmp_path / "im.gguf"
    _write_imatrix(p, [
        ("blk.0.attn_q.weight", np.array([2.0, 2.0], np.float32), 1.0),
        ("blk.1.ffn_up_exps.weight", np.array([8.0, 8.0, 8.0], np.float32), 4.0),
    ])
    vecs = read_imatrix_vectors(p)
    assert set(vecs) == {"blk.0.attn_q.weight", "blk.1.ffn_up_exps.weight"}
    np.testing.assert_allclose(vecs["blk.1.ffn_up_exps.weight"], np.full(3, 2.0))


def test_calibration_identity_records_hash_and_count(tmp_path):
    p = tmp_path / "im.gguf"
    _write_imatrix(p, [("blk.0.attn_q.weight", np.array([1.0, 1.0], np.float32), 1.0)])
    vecs, identity = imatrix_calibration(p)
    assert identity["kind"] == "gguf_imatrix"
    assert identity["name"] == "im.gguf"
    assert identity["key_count"] == 1
    assert identity["size_bytes"] == p.stat().st_size
    assert identity["sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
    assert identity["sampling"] == "per_channel_in_sum2_over_counts"
    # imatrix_calibration returns the same vectors read_imatrix_vectors does
    assert set(vecs) == set(read_imatrix_vectors(p))


def test_reads_legacy_dat_vectors_and_identity(tmp_path):
    p = tmp_path / "im.dat"
    _write_legacy_imatrix(
        p,
        [("blk.0.ffn_down.weight", 2, np.array([4.0, 8.0], np.float32))],
        chunks=7,
        dataset="calib.txt",
    )
    vecs, identity = imatrix_calibration(p)
    np.testing.assert_allclose(vecs["blk.0.ffn_down.weight"], [2.0, 4.0])
    assert identity["kind"] == "legacy_imatrix"
    assert identity["name"] == "im.dat"
    assert identity["key_count"] == 1
    assert identity["sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()


def test_legacy_dat_reader_streams_without_whole_file_read(tmp_path, monkeypatch):
    p = tmp_path / "im.dat"
    _write_legacy_imatrix(
        p,
        [("blk.0.ffn_down.weight", 2, np.array([4.0, 8.0], np.float32))],
    )

    original_read_bytes = type(p).read_bytes

    def read_bytes(self):
        if self == p:
            raise AssertionError("legacy imatrix reader must stream entries")
        return original_read_bytes(self)

    monkeypatch.setattr(type(p), "read_bytes", read_bytes)

    vecs = read_imatrix_vectors(p)
    np.testing.assert_allclose(vecs["blk.0.ffn_down.weight"], [2.0, 4.0])


def test_legacy_dat_ds4_expert_entries_collapse_to_input_vector(tmp_path):
    p = tmp_path / "ds4.dat"
    values = np.concatenate([
        np.full(4096, 2.0, np.float32),
        np.full(4096, 6.0, np.float32),
    ])
    _write_legacy_imatrix(
        p,
        [("blk.0.ffn_gate_exps.weight", 1, values)],
    )
    vecs = read_imatrix_vectors(p)
    assert vecs["blk.0.ffn_gate_exps.weight"].shape == (4096,)
    np.testing.assert_allclose(vecs["blk.0.ffn_gate_exps.weight"], 4.0)


def test_legacy_dat_has_no_expert_count_hotlist_data(tmp_path):
    p = tmp_path / "ds4.dat"
    _write_legacy_imatrix(
        p,
        [("blk.0.ffn_gate_exps.weight", 1, np.ones(4096, np.float32))],
    )
    assert imatrix_expert_counts(p) == {}


def test_invalid_imatrix_file_fails_with_clear_format_error(tmp_path):
    p = tmp_path / "bad.bin"
    p.write_bytes(b"not an imatrix")
    with pytest.raises(ValueError, match="unsupported imatrix format"):
        read_imatrix_vectors(p)


def test_identity_is_deterministic_for_same_file(tmp_path):
    p = tmp_path / "im.gguf"
    _write_imatrix(p, [("blk.0.attn_q.weight", np.array([3.0], np.float32), 1.0)])
    a = calibration_identity(p, {"blk.0.attn_q.weight": np.array([3.0])})
    b = calibration_identity(p, {"blk.0.attn_q.weight": np.array([3.0])})
    assert a == b


def test_stacked_expert_2d_statistics_aggregate_across_experts(tmp_path):
    """Expert tensors store in_sum2 [n_exp, in] + per-expert counts; importance
    must be the aggregate across the calibration corpus."""
    p = tmp_path / "im.gguf"
    # GGUF dims innermost-first: logical [n_exp=2, in=4] writes dims [4, 2]
    sum2 = np.array([[4.0, 8.0, 12.0, 16.0],
                     [1.0, 1.0, 1.0, 1.0]], np.float32)
    counts = np.array([[3.0, 1.0]], np.float32)
    tensors = [("blk.0.ffn_gate_exps.weight.in_sum2", sum2),
               ("blk.0.ffn_gate_exps.weight.counts", counts)]
    header = struct.pack("<IIQQ", _GGUF_MAGIC, 3, len(tensors), 0)
    infos = bytearray()
    offset = 0
    for name, arr in tensors:
        infos += _gguf_string(name.encode())
        infos += struct.pack("<I", arr.ndim)
        for d in reversed(arr.shape):  # innermost-first
            infos += struct.pack("<Q", d)
        infos += struct.pack("<I", _F32)
        infos += struct.pack("<Q", offset)
        offset += arr.nbytes
    head = header + bytes(infos)
    pad = (-len(head)) % 32
    with open(p, "wb") as f:
        f.write(head)
        f.write(b"\x00" * pad)
        for _n, arr in tensors:
            f.write(arr.tobytes())

    vecs = read_imatrix_vectors(p)
    # h = (row0 + row1) / (3 + 1)
    np.testing.assert_allclose(
        vecs["blk.0.ffn_gate_exps.weight"],
        (sum2[0] + sum2[1]) / 4.0)
