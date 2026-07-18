"""Allocation health gate: pure-function specs.

The gate rejects serve-unviable allocations the optimization proxy cannot see: a
collapsed backbone (most affine at <=3 bit), a critical single tensor below its floor
(lm_head/embed), or routed experts collapsed to 1-bit. These are exactly the collapse
signatures a quality/tau recipe can produce while still "satisfying" its objective.
"""

from __future__ import annotations

from moespresso.optimize.health import health_check


def _affine(name, bits, role=None):
    return {"kind": "affine", "source_name": name, "role": role or name, "bits": bits,
            "group_size": 64}


def _expert(name, bits):
    return {"kind": "expert", "source_name": name, "role": "moe.expert.gate", "bits": bits}


def _ds4_expert(name, bits, layer=0):
    return {
        "kind": "expert",
        "source_name": name,
        "role": "moe.expert.gate",
        "bits": bits,
        "layer_index": layer,
        "source_codec": "fp4_e2m1_ue8m0",
        "codec": "tq" if bits < 4 else "mxfp4",
    }


def _healthy_backbone(n=20, bits=8):
    return [_affine(f"L{i}.attn.q_proj", bits, role="attn.q_proj") for i in range(n)]


# --- healthy passes (no false positives) ---

def test_healthy_pg17_like_allocation_passes():
    # Reference shape: lm_head 6b, embed 4b, backbone high-bit, one stray 3-bit tensor.
    alloc = (_healthy_backbone(20, 8)
             + [_affine("lm_head", 6, role="lm_head"),
                _affine("embed", 4, role="embed_tokens"),
                _affine("L0.down", 3, role="ffn.down_proj")]  # the lone low-bit tensor
             + [_expert(f"e{i}.gate", 4) for i in range(10)])
    assert health_check(alloc) == []


# --- collapse signatures are rejected ---

def test_collapsed_backbone_is_rejected():
    # The --target-quality 0.95 --tau 0.9 signature: most affine at 3-bit.
    alloc = ([_affine(f"L{i}.q", 3, role="attn.q_proj") for i in range(18)]
             + [_affine("lm_head", 6, role="lm_head"),
                _affine("embed", 4, role="embed_tokens")])
    codes = [v.code for v in health_check(alloc)]
    assert "optimize.collapsed_backbone" in codes


def test_lm_head_below_floor_is_rejected():
    alloc = _healthy_backbone(20, 8) + [_affine("lm_head", 3, role="lm_head"),
                                        _affine("embed", 4, role="embed_tokens")]
    v = health_check(alloc)
    assert any(x.code == "optimize.collapsed_critical_tensor" and x.blocking for x in v)


def test_embed_below_floor_is_rejected():
    alloc = _healthy_backbone(20, 8) + [_affine("lm_head", 6, role="lm_head"),
                                        _affine("embed", 2, role="embed_tokens")]
    assert any(x.code == "optimize.collapsed_critical_tensor" for x in health_check(alloc))


def test_collapsed_experts_are_rejected():
    alloc = (_healthy_backbone(20, 8)
             + [_affine("lm_head", 6, role="lm_head"), _affine("embed", 4, role="embed_tokens")]
             + [_expert(f"e{i}.gate", 1) for i in range(20)])  # all 1-bit (>= min-count)
    assert any(x.code == "optimize.collapsed_experts" for x in health_check(alloc))


def test_deepseek_v4_hash_routed_source_fp4_expert_floor_is_rejected():
    alloc = (_healthy_backbone(20, 8)
             + [_affine("lm_head", 6, role="lm_head"),
                _affine("embed", 4, role="embed_tokens")]
             + [_ds4_expert("layers.0.ffn.experts.gate", 1, layer=0),
                _ds4_expert("layers.3.ffn.experts.gate", 1, layer=3)])

    findings = health_check(alloc)

    assert any(
        x.code == "optimize.deepseek_v4_hash_expert_below_floor"
        and x.path == "/layers.0.ffn.experts.gate"
        and x.blocking
        for x in findings
    )
    assert not any(
        x.code == "optimize.deepseek_v4_hash_expert_below_floor"
        and x.path == "/layers.3.ffn.experts.gate"
        for x in findings
    )


def test_findings_are_blocking_errors():
    alloc = [_affine("lm_head", 2, role="lm_head")]
    v = health_check(alloc)
    assert v and all(x.severity == "error" and x.blocking for x in v)


def test_empty_allocation_is_vacuously_healthy():
    assert health_check([]) == []
