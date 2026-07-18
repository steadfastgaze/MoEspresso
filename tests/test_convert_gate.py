"""Convert-time correctness gate.

The gate runs the ladder rungs the family profile declares against the just-written
package. These tests pin the gate's decision behavior: a good real-pipeline
package passes; a contract-violating one blocks (and convert refuses unless
--allow-incomplete); an unprofiled family is skipped. Requires the runtime stack.
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("jang_tools.turboquant")

from moespresso.correctness.gate import run_convert_gate  # noqa: E402
from moespresso.inventory.architecture_profile import (  # noqa: E402
    profile_for,
    qwen3_5_moe_profile,
)
from moespresso.package.convert import CORRECTNESS_DIR, convert  # noqa: E402

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


def _tiny_model(d, model_type="qwen3_moe"):
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
    arch = dict(ARCH, model_type=model_type)
    (d / "config.json").write_text(json.dumps(arch))


def test_gate_passes_for_a_good_real_pipeline_package(tmp_path):
    # The anti-false-reject check: a package the real convert pipeline produced must pass
    # the gate (writes correctness evidence, convert returns the manifest, no raise).
    src = tmp_path / "src"
    _tiny_model(src)
    out = tmp_path / "pkg"
    manifest = convert(src, out, allow_uniform=True, target_quality=0.5, shard_size_gb=0.0)
    assert manifest["artifact_kind"] == "package_manifest"
    # evidence for every rung that ran is persisted next to the package
    ev_dir = out / CORRECTNESS_DIR
    for rung in ("L0", "L0b", "L1", "L2"):
        assert (ev_dir / f"{rung}_evidence.json").exists(), f"missing {rung} evidence"


def test_gate_blocks_a_contract_violating_package(tmp_path):
    # The gate must refuse a package whose manifest violates the profile contract: here a
    # tensor declared by the inventory is dropped from the manifest (tensor_not_carried).
    src = tmp_path / "src"
    _tiny_model(src)
    out = tmp_path / "pkg"
    convert(src, out, allow_uniform=True, target_quality=0.5, shard_size_gb=0.0)

    inv = json.loads((out / "source_inventory.json").read_text())
    man = json.loads((out / "package_manifest.json").read_text())
    man["tensors"] = []  # drop everything the package was supposed to carry

    result = run_convert_gate(profile_for(ARCH), inv, man, src, out,
                              subject={"source_root": str(src)})
    assert not result.passed
    assert any(code == "correctness.tensor_not_carried" for _r, code, _m in result.blocking)


def test_convert_skips_gate_for_unprofiled_family(tmp_path, capsys):
    # Format-agnostic: a family with no registered profile must not be blocked. The gate is
    # skipped (loud warning) and the package is still written.
    src = tmp_path / "src"
    _tiny_model(src, model_type="some_new_arch_v9")
    out = tmp_path / "pkg"
    manifest = convert(src, out, allow_uniform=True, target_quality=0.5, shard_size_gb=0.0,
                       verbose=True)
    assert manifest["artifact_kind"] == "package_manifest"
    assert "correctness gate SKIPPED" in capsys.readouterr().out
    assert not (out / CORRECTNESS_DIR).exists()  # gate did not run -> no evidence dir


def test_full_attention_only_package_does_not_block_on_absent_conv1d(tmp_path):
    # A full-attention-only model legitimately has no conv1d. The profile requires the
    # conv1d/norm-shift coupling for the family, but expect_conv1d is False here, so an
    # absent conv1d must not block (the gate would otherwise false-reject a valid subset).
    src = tmp_path / "src"
    _tiny_model(src)  # layer_types = ["full_attention"]
    out = tmp_path / "pkg"
    # convert() itself would raise if the gate blocked; reaching a manifest proves it didn't.
    manifest = convert(src, out, allow_uniform=True, target_quality=0.5, shard_size_gb=0.0)
    l0b = json.loads((out / CORRECTNESS_DIR / "L0b_evidence.json").read_text())
    assert l0b["status"] == "valid"
    assert manifest["artifact_kind"] == "package_manifest"
    # sanity: the family did resolve (gate ran), it just didn't demand conv1d
    assert profile_for(ARCH)["family"] == qwen3_5_moe_profile()["family"]
