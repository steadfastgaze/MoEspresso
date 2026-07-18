"""Allocation health gate: a pure, model-free check on a rendered allocation.

The optimizer's objective (importance-weighted fidelity, plus an optional CVaR worst-layer
tail) can be satisfied by an allocation that nonetheless serves garbage: a quality target
met at the bit floor, or `tau` satisfied while most of the backbone sits at 3-bit (tail >=
tau yet hundreds of affine tensors at 3-bit). The serve cost of that is not visible to a
reconstruction-error proxy. This gate inspects the chosen bits and rejects the known-bad
signatures before a multi-GB package is written.

Pure: a list of allocation dicts in -> a list of Validation out (empty == healthy). No
model, no logits, no I/O. The caller (decide) appends these to the decision artifact's
validation and flips status to invalid on any blocking entry; convert offers an explicit
--allow-unhealthy escape hatch.

Thresholds come from known-coherent reference allocations, which carry roughly one
low-bit affine tensor.
"""

from __future__ import annotations

from moespresso.core.artifact import Validation

# Serve-critical single tensors: below this they corrupt the whole forward (the CVaR
# tail can't see them; each is its own 1-unit layer). Matches the serve-safe floors.
_CRITICAL_MIN_BITS = {"lm_head": 6, "embed_tokens": 4}

# A reconstruction proxy reads 3-bit affine as "fine", but a backbone mostly at <=3 bit
# serves garbage. The proven allocations have ~1 such tensor; allow a small margin, then
# reject. Expressed as a fraction of the affine backbone so it scales with model size.
# The fraction is only meaningful over a real backbone (hundreds of tensors); below
# _MIN_AFFINE_FOR_FRACTION the check is skipped so it can't false-positive on tiny
# synthetic allocations (the critical-tensor + expert checks still apply).
_LOW_BIT_AFFINE = 3
_MAX_LOW_BIT_AFFINE_FRACTION = 0.10
_MIN_AFFINE_FOR_FRACTION = 16

# Experts heavily at 1-bit is a known catastrophe (an early collapse mode). Routed
# experts may legitimately use low bits, but an all-1-bit majority is unsupported. As
# with the backbone, the fraction is only meaningful over a real expert set (a 35B MoE
# has ~120 groups); below _MIN_EXPERTS_FOR_FRACTION the check is skipped.
_MAX_ONE_BIT_EXPERT_FRACTION = 0.50
_MIN_EXPERTS_FOR_FRACTION = 16

# DeepSeek-V4 source-FP4 routed experts in the first three layers are hash-routed
# by token id. They are not protected by learned top-k route averaging, so the
# reconstruction proxy must not strand them below the lossless 4-bit tier.
_DEEPSEEK_V4_SOURCE_FP4_CODEC = "fp4_e2m1_ue8m0"
_DEEPSEEK_V4_HASH_ROUTED_LAYERS = 3
_DEEPSEEK_V4_HASH_EXPERT_MIN_BITS = 4


def health_check(allocation: list[dict]) -> list[Validation]:
    """Return blocking Validation entries for unsupported allocation signatures.

    `allocation` is the rendered per-tensor list from the decision: each entry has
    `kind` ("affine" | "expert" | "fp16_passthrough"), `role`, and (for affine/expert)
    `bits`. Empty result == the allocation is serve-viable by these structural checks.
    """
    out: list[Validation] = []
    affine = [a for a in allocation if a.get("kind") == "affine"]
    experts = [a for a in allocation if a.get("kind") == "expert"]

    # 1. Serve-critical single tensors below their floor.
    for a in affine:
        floor = _CRITICAL_MIN_BITS.get(a.get("role"))
        if floor is not None and a.get("bits", 0) < floor:
            out.append(Validation(
                "error", "optimize.collapsed_critical_tensor",
                f"{a['role']} at {a.get('bits')}-bit < serve-safe floor {floor}-bit "
                f"(serves garbage; the worst-layer tail cannot protect a single global tensor)",
                path=f"/{a.get('source_name', a.get('role'))}", phase="optimize",
                blocking=True, expected=floor, actual=a.get("bits")))

    # 2. Too much of the affine backbone at <= _LOW_BIT_AFFINE bits. Only meaningful
    # over a real backbone, so skip the fraction check on tiny allocations.
    if len(affine) >= _MIN_AFFINE_FOR_FRACTION:
        n_low = sum(1 for a in affine if a.get("bits", 0) <= _LOW_BIT_AFFINE)
        frac = n_low / len(affine)
        if frac > _MAX_LOW_BIT_AFFINE_FRACTION:
            out.append(Validation(
                "error", "optimize.collapsed_backbone",
                f"{n_low}/{len(affine)} affine tensors at <={_LOW_BIT_AFFINE}-bit "
                f"({frac:.0%} > {_MAX_LOW_BIT_AFFINE_FRACTION:.0%}); the backbone has "
                f"collapsed to low precision (a quality/tau target met at the floor)",
                path="/allocation", phase="optimize", blocking=True,
                expected=f"<={_MAX_LOW_BIT_AFFINE_FRACTION:.0%} at <={_LOW_BIT_AFFINE}-bit",
                actual=f"{frac:.0%}"))

    # 3. Routed experts overwhelmingly at 1-bit. Fraction only meaningful over a real
    # expert set, so skip on tiny allocations.
    if len(experts) >= _MIN_EXPERTS_FOR_FRACTION:
        n_one = sum(1 for a in experts if a.get("bits", 0) <= 1)
        frac = n_one / len(experts)
        if frac > _MAX_ONE_BIT_EXPERT_FRACTION:
            out.append(Validation(
                "error", "optimize.collapsed_experts",
                f"{n_one}/{len(experts)} expert groups at <=1-bit ({frac:.0%} > "
                f"{_MAX_ONE_BIT_EXPERT_FRACTION:.0%}); routed-expert precision has collapsed",
                path="/allocation", phase="optimize", blocking=True,
                expected=f"<={_MAX_ONE_BIT_EXPERT_FRACTION:.0%} at 1-bit",
                actual=f"{frac:.0%}"))

    for a in experts:
        if a.get("source_codec") != _DEEPSEEK_V4_SOURCE_FP4_CODEC:
            continue
        layer = a.get("layer_index")
        if not isinstance(layer, int) or layer >= _DEEPSEEK_V4_HASH_ROUTED_LAYERS:
            continue
        bits = a.get("bits", 0)
        if bits < _DEEPSEEK_V4_HASH_EXPERT_MIN_BITS:
            out.append(Validation(
                "error", "optimize.deepseek_v4_hash_expert_below_floor",
                f"{a.get('source_name', a.get('role'))} is a DeepSeek-V4 source-FP4 "
                f"expert in hash-routed layer {layer} but is allocated {bits}-bit; "
                f"hash-routed DS4 experts must stay at the lossless 4-bit tier",
                path=f"/{a.get('source_name', a.get('role'))}",
                phase="optimize",
                blocking=True,
                expected=_DEEPSEEK_V4_HASH_EXPERT_MIN_BITS,
                actual=bits))

    return out
