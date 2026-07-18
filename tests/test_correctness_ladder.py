"""Correctness ladder: L0 static contract + L1 norm-shift storage/runtime check.

The headline test is the regression: a package that stored conv1d pre-transposed
([out,k,1]) suppresses the runtime RMSNorm +1.0 shift the profile requires -> L1 must
block. The good package (conv1d source [out,1,k], norms stored unshifted) must
pass: L1 must not demand shifted norms at storage (the false-reject trap). All checks are
testable without a GPU/model (header shapes + profile), runnable anywhere.
"""

from __future__ import annotations

import json
import struct

import numpy as np

from moespresso.core.artifact import validate_base
from moespresso.correctness.ladder import (
    l0_static_contract,
    l0b_norm_shift_contract,
    make_correctness_evidence,
)
from moespresso.inventory.architecture_profile import (
    deepseek_v4_flash_profile,
    qwen3_5_moe_profile,
)


def _write_safetensors(path, tensors):
    header, blob, off = {}, bytearray(), 0
    for name, arr in tensors.items():
        a = np.ascontiguousarray(arr, dtype=np.float16)
        b = a.tobytes()
        header[name] = {"dtype": "F16", "shape": list(a.shape),
                        "data_offsets": [off, off + len(b)]}
        blob += b
        off += len(b)
    hjson = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        f.write(blob)


def _pkg(tmp_path, conv_shape):
    """A minimal package dir: one conv1d (given shape) + one norm, fp16."""
    d = tmp_path / "pkg"
    d.mkdir()
    rng = np.random.default_rng(0)
    _write_safetensors(d / "model-00001-of-00001.safetensors", {
        "language_model.model.layers.0.linear_attn.conv1d.weight":
            rng.standard_normal(conv_shape).astype(np.float16),
        "language_model.model.layers.0.input_layernorm.weight":
            rng.standard_normal((128,)).astype(np.float16),
    })
    return d


# --- L0b: the conv1d/norm-shift storage-vs-runtime contract ---

def test_l0b_blocks_pre_v8_conv1d_pretransposed(tmp_path):
    # conv1d stored [out,k,1] (last dim == 1) -> trigger absent -> the required norm shift
    # is suppressed at runtime -> norms ~1.0 too low -> garbage. L1 must block this.
    d = _pkg(tmp_path, (8192, 4, 1))
    v = l0b_norm_shift_contract(qwen3_5_moe_profile(), d)
    bad = [x for x in v if x.code == "correctness.norm_shift_suppressed"]
    assert bad and all(x.blocking for x in bad)
    assert {x.phase for x in bad} == {"L0b"}


def test_l0b_passes_good_package(tmp_path):
    d = _pkg(tmp_path, (8192, 1, 4))   # source shape -> trigger present
    v = l0b_norm_shift_contract(qwen3_5_moe_profile(), d)
    assert not any(x.blocking for x in v)


def test_l0b_blocks_when_required_conv1d_absent(tmp_path):
    # The profile requires the conv1d/norm-shift coupling. If no conv1d tensor exists in
    # the package, L1 cannot prove the runtime shift trigger and must block.
    d = tmp_path / "pkg"
    d.mkdir()
    _write_safetensors(d / "model-00001-of-00001.safetensors", {
        "language_model.model.layers.0.input_layernorm.weight":
            np.zeros((128,), dtype=np.float16)})   # norms present, conv1d absent
    v = l0b_norm_shift_contract(qwen3_5_moe_profile(), d)
    assert any(x.code == "correctness.required_conv1d_absent" and x.blocking for x in v)


def test_l0b_absent_conv1d_does_not_block_when_not_expected(tmp_path):
    # A full-attention-only / smoke package legitimately has no conv1d. With
    # expect_conv1d=False, an absent conv1d must not block (the family still requires the
    # coupling, but this package isn't expected to carry one).
    d = tmp_path / "pkg"
    d.mkdir()
    _write_safetensors(d / "model-00001-of-00001.safetensors", {
        "language_model.model.layers.0.input_layernorm.weight":
            np.zeros((128,), dtype=np.float16)})   # norms present, conv1d absent
    v = l0b_norm_shift_contract(qwen3_5_moe_profile(), d, expect_conv1d=False)
    assert not any(x.blocking for x in v)


def test_l0b_present_conv1d_is_shape_checked_even_when_not_expected(tmp_path):
    # expect_conv1d=False relaxes only the absence rule; a conv1d that is present is still
    # shape-checked, so the [out,k,1] storage bug can't hide in a partial/smoke package.
    d = _pkg(tmp_path, (8192, 4, 1))   # bad stored layout, last dim == 1
    v = l0b_norm_shift_contract(qwen3_5_moe_profile(), d, expect_conv1d=False)
    assert any(x.code == "correctness.norm_shift_suppressed" and x.blocking for x in v)


# --- L0: static contract (pure, no tensor IO) ---

def _inventory(tensors):
    return {"tensors": tensors}


def _manifest(entries):
    return {"tensors": entries}


def test_l0_blocks_unowned_source_tensor(tmp_path):
    # A source tensor with a role the profile doesn't assign a quant owner -> blocking.
    prof = qwen3_5_moe_profile()
    inv = _inventory([{"source_name": "x.weird_proj.weight", "role": "weird.unknown",
                       "kind": "affine"}])
    man = _manifest([{"source_name": "x.weird_proj.weight", "format": "affine"}])
    v = l0_static_contract(prof, inv, man)
    assert any(x.code == "correctness.unowned_tensor" and x.blocking for x in v)


def test_l0_blocks_quant_kind_mismatch_vs_profile():
    # Profile says ssm.in_proj_a is affine; a manifest that stored it as fp16 violates
    # the declared contract (this is exactly the in_proj bug class).
    prof = qwen3_5_moe_profile()
    inv = _inventory([{"source_name": "L0.linear_attn.in_proj_a.weight",
                       "role": "ssm.in_proj_a", "kind": "affine"}])
    man = _manifest([{"source_name": "L0.linear_attn.in_proj_a.weight", "format": "fp16"}])
    v = l0_static_contract(prof, inv, man)
    assert any(x.code == "correctness.quant_kind_mismatch" and x.blocking for x in v)


def test_l0_passes_contract_consistent_inventory():
    prof = qwen3_5_moe_profile()
    inv = _inventory([{"source_name": "L0.self_attn.q_proj.weight",
                       "role": "attn.q_proj", "kind": "affine"}])
    man = _manifest([{"source_name": "L0.self_attn.q_proj.weight", "format": "affine"}])
    assert not any(x.blocking for x in l0_static_contract(prof, inv, man))


def test_l0_treats_structural_passthrough_as_owned():
    # Norms/SSM-state are passthrough (kind=passthrough, structural role), stored fp16.
    # They form an explicit owner class that L0 must accept. A real package has
    # ~221 such tensors; flagging them false-rejects a proven-good package.
    prof = qwen3_5_moe_profile()
    inv = _inventory([
        {"source_name": "L0.input_layernorm.weight", "role": "norm.input_layernorm",
         "kind": "passthrough"},
        {"source_name": "L0.linear_attn.conv1d.weight", "role": "ssm.conv1d",
         "kind": "passthrough"},
    ])
    man = _manifest([
        {"source_name": "L0.input_layernorm.weight", "format": "fp16"},
        {"source_name": "L0.linear_attn.conv1d.weight", "format": "fp16"},
    ])
    assert not any(x.blocking for x in l0_static_contract(prof, inv, man))


def test_l0_treats_deepseek_v4_codec_sources_as_owned_inputs():
    prof = deepseek_v4_flash_profile()
    inv = _inventory([
        {"source_name": "layers.0.ffn.experts.0.w1.weight",
         "role": "moe.expert.gate", "kind": "expert_source",
         "layer_index": 0, "expert_index": 0, "projection": "gate"},
        {"source_name": "layers.0.ffn.experts.0.w1.scale",
         "role": "moe.expert.gate", "kind": "codec_scale",
         "layer_index": 0, "expert_index": 0, "projection": "gate"},
        {"source_name": "layers.0.attn.wq_a.weight",
         "role": "attn.wq_a", "kind": "affine", "layer_index": 0},
        {"source_name": "layers.0.attn.wq_a.scale",
         "role": "attn.wq_a.scale", "kind": "codec_scale", "layer_index": 0},
        {"source_name": "layers.0.attn.attn_sink",
         "role": "attn.attn_sink", "kind": "passthrough", "layer_index": 0},
    ])
    man = _manifest([
        {"source_name": "layers.0.ffn.experts.gate", "format": "tq",
         "projection": "gate", "layer_index": 0},
        {"source_name": "layers.0.attn.wq_a.weight", "format": "affine"},
        {"source_name": "layers.0.attn.attn_sink", "format": "raw_dtype_passthrough"},
    ])

    assert not any(x.blocking for x in l0_static_contract(prof, inv, man))


def test_l0_accepts_deepseek_v4_source_mxfp4_routed_expert_groups():
    prof = deepseek_v4_flash_profile()
    inv = _inventory([
        {"source_name": "layers.0.ffn.experts.0.w2.weight",
         "role": "moe.expert.down", "kind": "expert_source",
         "layer_index": 0, "expert_index": 0, "projection": "down"},
        {"source_name": "layers.0.ffn.experts.0.w2.scale",
         "role": "moe.expert.down", "kind": "codec_scale",
         "layer_index": 0, "expert_index": 0, "projection": "down"},
    ])
    man = _manifest([
        {"source_name": "layers.0.ffn.experts.down", "format": "mxfp4",
         "projection": "down", "layer_index": 0},
    ])

    assert not any(x.blocking for x in l0_static_contract(prof, inv, man))


def test_l0_blocks_required_tensor_missing_from_manifest():
    # A required (owned, non-excluded) source tensor that the manifest does not carry is
    # a contract violation that L0 must block (the omission bug class).
    prof = qwen3_5_moe_profile()
    inv = _inventory([{"source_name": "L0.self_attn.q_proj.weight",
                       "role": "attn.q_proj", "kind": "affine"}])
    man = _manifest([])   # q_proj omitted from the package
    v = l0_static_contract(prof, inv, man)
    assert any(x.code == "correctness.tensor_not_carried" and x.blocking for x in v)


def test_l0_blocks_manifest_tensor_not_in_inventory():
    # Reverse direction: a manifest carrying a tensor the inventory never declared must
    # block as undeclared unless the namespace is excluded. (An excluded-namespace
    # leak is a different code (excluded_tensor_carried), covered separately.)
    prof = qwen3_5_moe_profile()
    inv = _inventory([{"source_name": "L0.self_attn.q_proj.weight",
                       "role": "attn.q_proj", "kind": "affine"}])
    man = _manifest([
        {"source_name": "L0.self_attn.q_proj.weight", "format": "affine"},
        {"source_name": "ghost.weight", "format": "fp16"},
    ])
    v = l0_static_contract(prof, inv, man)
    undeclared = [x for x in v if x.code == "correctness.undeclared_package_tensor"]
    assert len(undeclared) == 1 and all(x.blocking for x in undeclared)
    assert undeclared[0].path == "/ghost.weight"


def test_l0_blocks_excluded_namespace_tensor_even_if_in_inventory():
    # Invariant: a package carrying a tensor in an excluded namespace blocks regardless of
    # whether inventory recorded it. Inventory currently omits vision/mtp, but if it later
    # records them as explicit "excluded" entries, the package must still not ship them.
    prof = qwen3_5_moe_profile()   # excludes "vision" and "mtp"
    inv = _inventory([
        {"source_name": "L0.self_attn.q_proj.weight", "role": "attn.q_proj", "kind": "affine"},
        {"source_name": "model.vision.blocks.0.attn.weight", "role": None, "kind": "excluded"},
        {"source_name": "mtp.layers.0.weight", "role": None, "kind": "excluded"},
    ])
    man = _manifest([
        {"source_name": "L0.self_attn.q_proj.weight", "format": "affine"},
        {"source_name": "model.vision.blocks.0.attn.weight", "format": "affine"},  # leaked
        {"source_name": "mtp.layers.0.weight", "format": "fp16"},                  # leaked
    ])
    v = l0_static_contract(prof, inv, man)
    leaked = [x for x in v if x.code == "correctness.excluded_tensor_carried"]
    assert len(leaked) == 2 and all(x.blocking for x in leaked)
    assert {x.path for x in leaked} == {"/model.vision.blocks.0.attn.weight",
                                        "/mtp.layers.0.weight"}


def test_l0_blocks_visual_namespace_tensor_in_inventory_without_crashing():
    # The real Qwen vision prefix is model.visual.* (not "vision"). An inventory that
    # records it kind='excluded', role=None must not crash L0 (role.startswith(None)) and
    # must block when carried. Regression: the profile token has to match model.visual.*.
    prof = qwen3_5_moe_profile()
    inv = _inventory([
        {"source_name": "L0.self_attn.q_proj.weight", "role": "attn.q_proj", "kind": "affine"},
        {"source_name": "model.visual.blocks.0.attn.qkv.weight", "role": None,
         "kind": "excluded"},
    ])
    man = _manifest([
        {"source_name": "L0.self_attn.q_proj.weight", "format": "affine"},
        {"source_name": "model.visual.blocks.0.attn.qkv.weight", "format": "affine"},  # leaked
    ])
    v = l0_static_contract(prof, inv, man)   # must not raise
    leaked = [x for x in v if x.code == "correctness.excluded_tensor_carried"]
    assert len(leaked) == 1 and leaked[0].blocking
    assert leaked[0].path == "/model.visual.blocks.0.attn.qkv.weight"


def test_l0_excluded_kind_not_carried_is_clean_not_unowned():
    # An inventory tensor marked kind='excluded' (role None) whose name matches no declared
    # namespace token, and that is absent from the package, remains out of scope. L0 must
    # neither require nor flag it. (Defensive: profile may miss a namespace.)
    prof = qwen3_5_moe_profile()
    inv = _inventory([
        {"source_name": "L0.self_attn.q_proj.weight", "role": "attn.q_proj", "kind": "affine"},
        {"source_name": "some.future.aux_head.weight", "role": None, "kind": "excluded"},
    ])
    man = _manifest([{"source_name": "L0.self_attn.q_proj.weight", "format": "affine"}])
    v = l0_static_contract(prof, inv, man)   # must not raise
    assert not any(x.blocking for x in v), [x.code for x in v if x.blocking]


def test_l0_blocks_carrying_a_tensor_the_inventory_marked_excluded():
    # The symmetric rule: if the inventory explicitly says a source tensor is out-of-scope
    # (kind='excluded'), carrying it must block, even if the profile never tokenized its
    # namespace. Inventory-declared exclusion remains authoritative even when the profile
    # omits that namespace.
    prof = qwen3_5_moe_profile()
    inv = _inventory([
        {"source_name": "L0.self_attn.q_proj.weight", "role": "attn.q_proj", "kind": "affine"},
        {"source_name": "some.future.aux_head.weight", "role": None, "kind": "excluded"},
    ])
    man = _manifest([
        {"source_name": "L0.self_attn.q_proj.weight", "format": "affine"},
        {"source_name": "some.future.aux_head.weight", "format": "fp16"},   # carried anyway
    ])
    v = l0_static_contract(prof, inv, man)
    leaked = [x for x in v if x.code == "correctness.excluded_tensor_carried"]
    assert len(leaked) == 1 and leaked[0].blocking
    assert leaked[0].path == "/some.future.aux_head.weight"


def test_l0_validates_every_duplicate_manifest_entry_format():
    # Fused gate_up_proj legitimately appears twice under one source_name (projection
    # gate + up). L0 must validate each entry's format: collapsing to the last format
    # per source_name lets a bad duplicate hide behind a good one. Both orders must block.
    prof = qwen3_5_moe_profile()
    inv = _inventory([{"source_name": "L0.mlp.experts.gate_up_proj",
                       "role": "moe.expert.gate", "kind": "expert"}])
    bad_first = _manifest([
        {"source_name": "L0.mlp.experts.gate_up_proj", "format": "fp16",
         "projection": "gate"},                                    # wrong (should be tq)
        {"source_name": "L0.mlp.experts.gate_up_proj", "format": "tq",
         "projection": "up"},                                      # correct
    ])
    bad_last = _manifest([bad_first["tensors"][1], bad_first["tensors"][0]])  # reversed
    for man in (bad_first, bad_last):
        v = l0_static_contract(prof, inv, man)
        mism = [x for x in v if x.code == "correctness.quant_kind_mismatch"]
        assert mism and all(x.blocking for x in mism), "bad duplicate format must block"


def test_l0_blocks_entry_missing_source_name_instead_of_crashing():
    # A correctness gate must not crash on a malformed artifact: that's the same failure
    # class as the role=None crash. An inventory or manifest entry with no source_name is
    # itself a contract violation: block with correctness.malformed_entry, never KeyError.
    prof = qwen3_5_moe_profile()
    bad_man = l0_static_contract(prof, {"tensors": []}, {"tensors": [{"format": "affine"}]})
    assert any(x.code == "correctness.malformed_entry" and x.blocking for x in bad_man)
    bad_inv = l0_static_contract(prof, {"tensors": [{"role": "attn.q_proj",
                                                     "kind": "affine"}]}, {"tensors": []})
    assert any(x.code == "correctness.malformed_entry" and x.blocking for x in bad_inv)


def test_l0_on_empty_inputs_is_clean():
    # Degenerate but well-formed: no tensors anywhere -> no findings, no crash.
    assert l0_static_contract(qwen3_5_moe_profile(), {}, {}) == []


# --- evidence artifact ---

def test_correctness_evidence_is_a_valid_artifact(tmp_path):
    d = _pkg(tmp_path, (8192, 4, 1))   # the bad one -> blocking finding
    findings = l0b_norm_shift_contract(qwen3_5_moe_profile(), d)
    ev = make_correctness_evidence(
        {"source_root": "toy", "source_format": "hf_safetensors"},
        rung="L0b", findings=findings)
    assert ev["artifact_kind"] == "correctness_evidence"
    assert ev["artifact_id"].startswith("correct:")
    assert validate_base(ev) == []
    assert ev["status"] == "invalid"      # a blocking finding -> invalid evidence
    assert ev["rung"] == "L0b"
