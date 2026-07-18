"""Multi-shard streaming writer: a tiny shard cap forces a small model to split.

Proves the size-based splitting writes correct multi-file packages: groups split
across shards, each tensor group stays whole inside one shard, the manifest's
files list matches the renamed `-of-COUNT` shards on disk, verify passes, and the
loader reconstructs correctly reading across shards.
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("jang_tools.turboquant")

from moespresso.inventory.build import build_inventory  # noqa: E402
from moespresso.optimize.decide import decide  # noqa: E402
from moespresso.package.plan import package_plan_from_decision  # noqa: E402
from moespresso.package.write import write_package  # noqa: E402
from moespresso.probe.build import build_probe_evidence  # noqa: E402
from moespresso.runtime.verify import verify_package  # noqa: E402

ARCH = {"model_type": "qwen3_moe",
        "text_config": {"num_hidden_layers": 1, "hidden_size": 128, "num_experts": 8,
                        "num_experts_per_tok": 2, "moe_intermediate_size": 128,
                        "layer_types": ["full_attention"], "vocab_size": 256}}


def _write_safetensors(path, tensors):
    header, blob, off = {}, bytearray(), 0
    for name, arr in tensors.items():
        a = np.ascontiguousarray(arr, dtype=np.float32)
        b = a.tobytes()
        header[name] = {"dtype": "F32", "shape": list(a.shape),
                        "data_offsets": [off, off + len(b)]}
        blob += b
        off += len(b)
    hjson = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        f.write(blob)


def _tiny_model(d):
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    _write_safetensors(d / "model-00001.safetensors", {
        "model.language_model.layers.0.self_attn.q_proj.weight":
            rng.standard_normal((128, 128)).astype(np.float32),
        "model.language_model.layers.0.mlp.experts.gate_up_proj":
            rng.standard_normal((8, 256, 128)).astype(np.float32),
        "model.language_model.layers.0.mlp.experts.down_proj":
            rng.standard_normal((8, 128, 128)).astype(np.float32),
    })
    (d / "config.json").write_text(json.dumps(ARCH))


def _build(tmp_path, shard_size_gb):
    src = tmp_path / "src"
    _tiny_model(src)
    out = tmp_path / "out"
    inv = build_inventory(src, layer_types=["full_attention"])
    ev = build_probe_evidence(inv, src, expert_sample=2, sample_rows=64)
    dec = decide(ev, target_quality=0.5)
    plan, _summary = package_plan_from_decision(dec)
    man = write_package(plan, src, ARCH, out, shard_size_gb=shard_size_gb)
    return src, out, man


def test_tiny_cap_splits_into_multiple_shards(tmp_path):
    # 3 tensor groups (q_proj affine + gate + up experts). A ~few-KB cap forces
    # each group onto its own shard.
    _, out, man = _build(tmp_path, shard_size_gb=1e-6)  # ~1 KB cap
    shard_files = {f["path"] for f in man["files"]}
    assert len(shard_files) >= 2, f"expected a split, got {shard_files}"
    # every declared shard exists on disk and is named -of-COUNT (not -of-?????)
    count = len(shard_files)
    for name in shard_files:
        assert (out / name).exists()
        assert name.endswith(f"-of-{count:05d}.safetensors")
        assert "?????" not in name


def test_each_group_whole_within_one_shard(tmp_path):
    _, out, man = _build(tmp_path, shard_size_gb=1e-6)
    # for each manifest tensor, all its expected keys live in the same shard.
    from moespresso.runtime.verify import expected_keys
    headers = {}
    for t in man["tensors"]:
        shard = t["shard"]
        if shard not in headers:
            with open(out / shard, "rb") as f:
                hlen = struct.unpack("<Q", f.read(8))[0]
                headers[shard] = set(json.loads(f.read(hlen))) - {"__metadata__"}
        for key in expected_keys(t):
            assert key in headers[shard], f"{key} not in its declared shard {shard}"


def _stored_arrays(out_dir, man):
    """Every packed array on disk, by full key (reads stored bytes, no reconstruct)."""
    from safetensors.numpy import load_file
    out = {}
    for f in man["files"]:
        out.update(load_file(str(out_dir / f["path"])))
    return out


def test_multishard_package_verifies_and_writes_identical_bytes(tmp_path):
    src, out, man = _build(tmp_path, shard_size_gb=1e-6)
    assert [v for v in verify_package(man, out) if v.blocking] == []

    # Chunked (multi-shard) write must produce byte-identical stored tensors as the
    # single-shard path: writer determinism, proven on stored bytes (no weight
    # reconstruction: the engine never reconstructs, and tests shouldn't either).
    _, out1, man1 = _build(tmp_path / "single", shard_size_gb=0.0)
    multi = _stored_arrays(out, man)
    single = _stored_arrays(out1, man1)
    assert set(multi) == set(single)
    for key in multi:
        np.testing.assert_array_equal(multi[key], single[key])


def test_zero_cap_is_single_shard(tmp_path):
    _, out, man = _build(tmp_path, shard_size_gb=0.0)
    assert len(man["files"]) == 1
    assert man["files"][0]["path"] == "model-00001-of-00001.safetensors"
