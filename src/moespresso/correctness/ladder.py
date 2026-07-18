"""Correctness ladder rungs L0 (static contract) + L0b (header storage contract).

Checks run without a GPU or model and confirm that a package satisfies its family's
`architecture_profile` contract. They emit standalone `correctness_evidence` and stay
outside convert, serve, and verify.

L0 (pure, no tensor IO): every source tensor has a declared quant owner in the profile,
and the manifest's chosen format matches that owner (catches the in_proj-as-fp16 class).

L0b (header-only, no weight bytes, no model): the conv1d/norm-shift storage-vs-runtime
contract. The profile declares norms are stored unshifted and the runtime sanitizer adds
+delta only when the conv1d trigger holds. So this rung must not demand shifted norms at
storage (that would reject a correct package); it checks the storage shape satisfies the
trigger, and blocks only when a required runtime shift is suppressed because the trigger
is absent on disk (conv1d stored [out,k,1], last dim == 1, suppresses the shift so norms
load too low and the model emits garbage).
"""

from __future__ import annotations

from pathlib import Path

from moespresso.core.artifact import Validation, make_artifact
from moespresso.inventory.safetensors_header import read_header

PRODUCER = {"tool": "moespresso.correctness", "version": "1.0.0"}

# Map manifest on-disk format -> the profile's quant-owner vocabulary.
_FORMAT_TO_OWNER = {
    "affine": "affine",
    "mxfp4": "affine",
    "mxfp8": "affine",
    "tq": "tq",
    "fp16": "fp16",
}
_ROUTED_EXPERT_FORMATS = {"tq", "mxfp4"}


def _role_owner(profile: dict, role: str) -> str | None:
    """Declared quant owner for a typed role, or None if the profile doesn't assign one.

    Expert roles are `moe.expert.<proj>`; the profile keys them under `moe.expert`.
    """
    if not role:
        return None  # untyped tensor (e.g. kind='excluded' carries role=None): no owner
    rq = profile.get("role_quant", {})
    if role in rq:
        return rq[role]
    if role.startswith("moe.expert"):
        return rq.get("moe.expert")
    return None


def _owner_class(profile: dict, tensor: dict):
    """Declared owner class for an inventory tensor: an affine/tq/fp16 quant owner,
    'passthrough' for a declared structural tensor, or None if unowned.

    Structural passthrough (norms, SSM state, conv1d) is its own owner class: the
    inventory marks it kind='passthrough' with a structural role, and the profile lists
    that role in `structural_passthrough`. It is carried verbatim in fp16.
    """
    role = tensor.get("role")
    kind = tensor.get("kind")
    if kind == "excluded":
        return "excluded"  # inventory marked it out-of-scope (role is None by convention)
    if kind in {"codec_scale", "expert_source"}:
        return "source_codec"
    if kind == "passthrough":
        role_owner = _role_owner(profile, role)
        if role_owner == "raw_dtype_passthrough":
            return role_owner
        if role in profile.get("structural_passthrough", []):
            return "passthrough"
        return None  # passthrough kind but the profile doesn't declare this role
    return _role_owner(profile, role)


def l0_static_contract(profile: dict, inventory: dict, manifest: dict) -> list[Validation]:
    """L0: every non-excluded source tensor is owned by the profile, carried in the
    package, and stored as its owner declares. Pure: inventory + manifest + profile only.

    Owner classes: a quant owner from `role_quant` (affine/tq/fp16), or 'passthrough' for a
    declared structural tensor (stored fp16). Excluded-namespace tensors are skipped.
    """
    out: list[Validation] = []
    excluded = profile.get("excluded_namespaces", {})

    def is_excluded(t: dict) -> bool:
        # Out-of-scope if the profile excludes its namespace or the inventory itself marked
        # it kind='excluded' (belt-and-suspenders: a kind='excluded' entry whose name the
        # profile forgot to list must still never be owned/required, and never crash).
        return (any(ns in t["source_name"] for ns in excluded)
                or t.get("kind") == "excluded")

    def malformed(t: dict, origin: str) -> bool:
        # Convert malformed artifacts into findings. A missing source_name is a
        # contract violation and must never escape as a KeyError.
        if t.get("source_name"):
            return False
        out.append(Validation(
            "error", "correctness.malformed_entry",
            f"a {origin} entry has no 'source_name' ({t!r}): a malformed package contract "
            f"cannot be validated",
            path="/", phase="L0", blocking=True))
        return True

    # Owner + carried checks are inventory-driven; owner_by_name lets the per-manifest-entry
    # format check below resolve each entry to its declared owner. excluded_by_name records
    # names the inventory declared out-of-scope (kind='excluded') so the manifest loop can
    # block carrying them even when the profile never tokenized the namespace.
    owner_by_name: dict[str, str | None] = {}
    inv_names: set[str] = set()
    excluded_by_name: set[str] = set()
    logical_names: dict[str, str] = {}
    for t in inventory.get("tensors", []):
        if malformed(t, "inventory"):
            continue
        name, role = t["source_name"], t.get("role")
        inv_names.add(name)
        if t.get("kind") == "expert_source" and t.get("layer_index") is not None:
            projection = t.get("projection")
            if projection:
                logical_names[
                    f"layers.{int(t['layer_index'])}.ffn.experts.{projection}"
                ] = "routed_expert"
        if is_excluded(t):
            excluded_by_name.add(name)
            continue  # explicitly excluded modality; outside this package's scope
        owner = _owner_class(profile, t)
        owner_by_name[name] = owner
        if owner is None:
            out.append(Validation(
                "error", "correctness.unowned_tensor",
                f"{name} (role {role!r}, kind {t.get('kind')!r}) has no declared owner in "
                f"the architecture_profile. Every tensor needs an explicit owner "
                f"(quantized / passthrough / excluded)",
                path=f"/{name}", phase="L0", blocking=True))
    owner_by_name.update(logical_names)
    inv_names.update(logical_names)

    present_names = {t["source_name"] for t in manifest.get("tensors", []) if t.get("source_name")}
    for t in inventory.get("tensors", []):
        name = t.get("source_name")
        if not name or is_excluded(t) or owner_by_name.get(name) is None:
            continue  # malformed already flagged above; excluded/unowned skip carriage
        if owner_by_name[name] == "source_codec":
            continue  # source-codec inputs are consumed into decoded package tensors
        # Owned tensors must be carried in the package (or they'd be silently dropped).
        if name not in present_names:
            out.append(Validation(
                "error", "correctness.tensor_not_carried",
                f"{name} (owner {owner_by_name[name]}) is owned but absent from the package "
                f"manifest: a required tensor is not carried",
                path=f"/{name}", phase="L0", blocking=True))

    # Per-entry checks over every manifest tensor (not collapsed by source_name): a fused
    # gate_up_proj appears twice (gate + up); each entry's stored format must match the
    # declared owner, so a bad duplicate can't hide behind a good one. Also the reverse
    # ownership direction: an entry whose source_name the inventory never declared (ghost
    # or excluded-modality leak like model.visual.*) is a contract violation.
    for t in manifest.get("tensors", []):
        if malformed(t, "manifest"):
            continue
        name, stored = t["source_name"], t.get("format")
        proj = t.get("projection")
        where = f"{name} (projection {proj!r})" if proj else name
        # An out-of-scope tensor must not be carried. Out-of-scope by either authority: the
        # profile excludes its namespace (profile-driven: catches a future inventory that
        # forgot to mark it), or the inventory itself declared it kind='excluded'
        # (inventory-driven: catches a name the profile never tokenized). Either is a
        # contract violation: a package shipping a tensor declared out-of-scope.
        ns = next((n for n in excluded if n in name), None)
        if ns is not None or name in excluded_by_name:
            why = (f"is in excluded namespace {ns!r}" if ns is not None
                   else "was declared out-of-scope (kind='excluded') by the inventory")
            out.append(Validation(
                "error", "correctness.excluded_tensor_carried",
                f"{where} {why} but is carried in the package manifest: an excluded tensor "
                f"must not ship",
                path=f"/{name}", phase="L0", blocking=True))
            continue
        if name not in inv_names:
            out.append(Validation(
                "error", "correctness.undeclared_package_tensor",
                f"{where} is carried in the package manifest but is not declared by the "
                f"source inventory: the package must not ship undeclared tensors",
                path=f"/{name}", phase="L0", blocking=True))
            continue
        owner = owner_by_name.get(name)
        if owner is None:
            continue  # already flagged unowned (or excluded) above
        if owner == "routed_expert":
            if stored not in _ROUTED_EXPERT_FORMATS:
                out.append(Validation(
                    "error", "correctness.quant_kind_mismatch",
                    f"{where} declared owner {owner!r} but stored as {stored!r}: package "
                    f"violates the family quant contract",
                    path=f"/{name}", phase="L0", blocking=True,
                    expected=sorted(_ROUTED_EXPERT_FORMATS), actual=stored))
            continue
        expected_fmt = "fp16" if owner == "passthrough" else owner
        if _FORMAT_TO_OWNER.get(stored, stored) != expected_fmt:
            out.append(Validation(
                "error", "correctness.quant_kind_mismatch",
                f"{where} declared owner {owner!r} but stored as {stored!r}: package "
                f"violates the family quant contract",
                path=f"/{name}", phase="L0", blocking=True,
                expected=expected_fmt, actual=stored))
    return out


def _conv1d_transforms(profile: dict):
    """(conv1d_layout, rmsnorm_shift) transform declarations, or (None, None)."""
    by_name = {t["name"]: t for t in profile.get("transforms", [])}
    return by_name.get("conv1d_layout"), by_name.get("rmsnorm_shift")


def l0b_norm_shift_contract(profile: dict, package_dir: Path, *,
                            expect_conv1d: bool = True) -> list[Validation]:
    """L0b: the conv1d/norm-shift storage-vs-runtime contract (header-only).

    Checks that the package's stored conv1d shape satisfies the sanitizer trigger the
    profile declares, so the required runtime RMSNorm shift will actually fire. Blocks
    when the shift is required but the on-disk conv1d shape suppresses the trigger (which
    makes norms load too low and the model emit garbage). Does not inspect norm values
    (they are stored unshifted by contract).

    `expect_conv1d` says whether this package should contain a conv1d at all: the coupling
    is `required` for the family, but a package whose layers are all full-attention (or a
    reduced/smoke build) legitimately has none. When False, an absent conv1d is fine (no
    block); a present one is still shape-checked (a suppressed-trigger shape can't hide in
    a subset). Default True preserves the strict standalone contract.
    """
    out: list[Validation] = []
    conv, norm = _conv1d_transforms(profile)
    if conv is None or norm is None or not norm.get("required"):
        return out  # family declares no required coupled shift -> nothing to enforce

    trigger = conv.get("sanitizer_trigger", {})
    suffix = trigger.get("tensor_suffix", "linear_attn.conv1d.weight")
    last_dim_not = trigger.get("last_dim_not")

    package_dir = Path(package_dir)
    shards = sorted(package_dir.glob("model-*.safetensors"))
    seen_conv = False
    for shard in shards:
        header = read_header(shard)
        for key, meta in header.items():
            if not key.endswith(suffix):
                continue
            seen_conv = True
            shape = meta.get("shape", [])
            # Trigger holds iff the stored shape's last dim != last_dim_not (e.g. != 1).
            trigger_present = bool(shape) and shape[-1] != last_dim_not
            if not trigger_present:
                out.append(Validation(
                    "error", "correctness.norm_shift_suppressed",
                    f"{key} stored shape {tuple(shape)} has last dim == {last_dim_not}, "
                    f"so the runtime sanitizer trigger ({conv['name']}) does NOT fire; the "
                    f"REQUIRED RMSNorm +{norm.get('delta')} shift ({norm['name']}) is "
                    f"suppressed -> norms load ~{norm.get('delta')} too low -> garbage. "
                    f"Store conv1d in source shape {conv.get('store_shape')!r} instead.",
                    path=f"/{key}", phase="L0b", blocking=True,
                    expected=f"last dim != {last_dim_not} (source [out,1,k])",
                    actual=f"last dim == {last_dim_not}"))
    if not seen_conv and expect_conv1d:
        # The profile requires the coupled shift (checked above via norm['required']) and
        # this package is expected to carry conv1d (it has linear-attention layers); if no
        # conv1d tensor exists, the trigger is unprovable -> the required shift can't be
        # established -> block. (A full-attention-only / smoke package sets expect_conv1d
        # False, so a legitimately-absent conv1d does not block.)
        out.append(Validation(
            "error", "correctness.required_conv1d_absent",
            f"profile requires the conv1d/norm-shift coupling ({norm['name']}) but no "
            f"'*{suffix}' tensor was found in the package shards: the required runtime "
            f"shift cannot be established",
            path="/", phase="L0b", blocking=True))
    return out


def make_correctness_evidence(subject: dict, *, rung: str, findings: list[Validation],
                              inputs: list | None = None) -> dict:
    """Wrap a rung's findings into a `correctness_evidence` artifact.

    status=invalid if any finding is blocking, else valid, so a bad package's evidence
    is unmistakably invalid, never silently "valid with notes".
    """
    blocking = any(f.blocking for f in findings)
    return make_artifact(
        "correctness_evidence", subject, PRODUCER,
        status="invalid" if blocking else "valid",
        validation=findings, inputs=inputs or [],
        rung=rung,
        summary={"findings": len(findings),
                 "blocking": sum(1 for f in findings if f.blocking)},
    )
