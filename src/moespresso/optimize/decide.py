"""Build an `optimizer_decision` artifact from a `probe_evidence` artifact.

The adapter (typed roles -> working dicts) + the numerical core (allocate.py) +
the artifact emitter. Reads structure only from the probe_evidence units' typed
fields (role, layer_index, projection, kind); never re-parses tensor names.

An infeasible run produces a durable, inspectable artifact with
`feasibility=infeasible` or `infeasible_under_budget` plus a blocking validation.
"""

from __future__ import annotations

from moespresso.core.artifact import Validation, make_artifact
from moespresso.optimize import allocate
from moespresso.optimize.aggregate import fidelity, worst_layer_tail
from moespresso.optimize.health import health_check
from moespresso.optimize.monotone import (
    monotone_envelope_by_bits,
    non_monotonic_inversions,
)
from moespresso.optimize.sizes import affine_bytes, mx_float_bytes

PRODUCER = {"tool": "moespresso.optimize", "version": "1.0.0"}

# Roles kept fp16 (never quantized): only the router / shared-expert gates. They
# select experts via argmax/top-k, a discrete decision a reconstruction-error proxy
# cannot score, so the optimizer must not assign them bits. Every other 2D non-expert
# weight (including the SSM in_proj_*) is quantized affine (the reference bundle
# stores in_proj_a/b at 8b/gs32). Do not add
# in_proj_a/b here: keeping them fp16 makes mjtq diverge from the reference bundle and serve garbage.
# Pinned by test_decide.test_ssm_in_proj_a_b_are_quantized_not_fp16.
FP16_ROLES = frozenset({
    "moe.router_gate", "moe.shared_expert_gate",
})

# Serve-critical safety floors (always applied, independent of tau). lm_head and
# embed_tokens are single global tensors (each its own 1-unit "layer") so the CVaR
# worst-layer tail cannot protect them; a quality/tau recipe otherwise strands them at
# the bit floor (the collapse signature where the worst-layer tail strands
# lm_head/embed at the bit floor). These are structural-safety constraints outside
# the optimization objective. The health gate protects the backbone; broad per-role
# floors would recreate
# the reverted uniform-high mistake).
_LM_HEAD_DEFAULT_BITS = 6
_SAFETY_MIN_BITS = {"embed_tokens": 4}  # lm_head handled via lm_head_bits (caller-tunable)

_LOSSLESS_MXFP4_SOURCE_CODEC = "fp4_e2m1_ue8m0"
_DEEPSEEK_V4_HASH_ROUTED_LAYERS = 3
_DEEPSEEK_V4_HASH_EXPERT_MIN_BITS = 4


def _layer_key(unit: dict):
    """Layer id for a non-expert unit; embed/lm_head/global become their own key."""
    li = unit.get("layer_index")
    return li if li is not None else unit["source_name"]


def _quality_table_expert(unit: dict) -> dict[int, float]:
    """{bits:int -> q} from a probe_evidence expert unit's string-keyed table.

    Projected onto the monotone-in-bits envelope so sampling-noise inversions
    (a higher bit measuring slightly worse) can't be exploited by the greedy.
    """
    return monotone_envelope_by_bits({int(k): v for k, v in unit["quality"].items()})


def _quality_table_affine(unit: dict) -> dict[tuple[int, int], float]:
    """{(bits,gs) -> q} from an affine unit's "bits_gs" string-keyed table.

    Projected onto the monotone-in-bits envelope per group_size (more bits
    never yields less q) so the greedy can't exploit a noise dip to pick fewer bits.
    """
    out: dict[tuple[int, int], float] = {}
    for key, val in unit["quality"].items():
        b, gs = key.split("_")
        out[(int(b), int(gs))] = val
    return monotone_envelope_by_bits(out)


def _quality_table_dense_mx(unit: dict) -> dict[tuple[str, int, int], float]:
    """{(format,bits,gs) -> q} from a unit's measured dense MX candidates.

    Projected onto the monotone-in-bits envelope per group_size, matching the
    affine/expert invariant that the optimizer never sees higher bits as lower
    quality due to sampling noise.
    """
    out: dict[tuple[str, int, int], float] = {}
    for key, val in unit.get("dense_codec_quality", {}).items():
        parts = str(key).split("_")
        if len(parts) != 3:
            continue
        fmt, bits, gs = parts
        if fmt not in allocate.DENSE_MX_FORMATS:
            continue
        out[(fmt, int(bits), int(gs))] = val
    return monotone_envelope_by_bits(out)


def _raw_quality_table(unit: dict):
    """Parse a unit's raw (pre-envelope) q-table, for inversion reporting only."""
    if unit["kind"] == "expert":
        return {int(k): v for k, v in unit["quality"].items()}
    out: dict[tuple[int, int], float] = {}
    for key, val in unit["quality"].items():
        b, gs = key.split("_")
        out[(int(b), int(gs))] = val
    return out


def _q_inversion_validations(evidence: dict) -> list[Validation]:
    """Warnings for units whose raw q-table dips beyond the noise band as bits rise.

    The monotone envelope smooths these for the allocation, but a genuine
    inversion is a measurement smell (re-measure), so we surface it, never hide it.
    Non-blocking: the envelope already made the allocation safe.
    """
    out: list[Validation] = []
    for u in evidence["units"]:
        if u["kind"] == "fp16_passthrough" or "quality" not in u:
            continue
        inversions = non_monotonic_inversions(_raw_quality_table(u))
        if inversions:
            worst = max(inversions, key=lambda x: x[2])
            out.append(Validation(
                "warning", "optimize.q_inversion",
                f"{u['source_name']}: q drops {worst[2]:.3f} from {worst[0]}-bit to "
                f"{worst[1]}-bit (beyond the noise band): likely a sampling artifact; "
                f"the envelope smoothed it for allocation, but consider re-measuring",
                path=f"/{u['source_name']}", phase="optimize", blocking=False))
    return out


def _validate_affine_role_weights(weights: dict[str, float] | None) -> dict[str, float]:
    if weights is None:
        return {}
    out = {}
    for role, value in weights.items():
        weight = float(value)
        if weight <= 0:
            raise ValueError(f"affine role weight for {role!r} must be > 0, got {value!r}")
        out[str(role)] = weight
    return dict(sorted(out.items()))


def _validate_affine_role_bit_weights(
    weights: dict[str, dict[int | str, float]] | None,
) -> dict[str, dict[int, float]]:
    if weights is None:
        return {}
    out = {}
    for role, bit_map in weights.items():
        clean = {}
        for bit, value in bit_map.items():
            bit_i = int(bit)
            if bit_i not in allocate.AFFINE_BITS:
                raise ValueError(f"affine role bit weight for {role!r} uses unsupported bit {bit!r}")
            weight = float(value)
            if weight <= 0:
                raise ValueError(
                    f"affine role bit weight for {role!r} at {bit_i}b must be > 0, "
                    f"got {value!r}")
            clean[bit_i] = weight
        out[str(role)] = dict(sorted(clean.items()))
    return dict(sorted(out.items()))


def _validate_affine_role_min_bits(weights: dict[str, int] | None) -> dict[str, int]:
    if weights is None:
        return {}
    out = {}
    for role, bits in weights.items():
        bit_i = int(bits)
        if bit_i not in allocate.AFFINE_BITS:
            raise ValueError(f"affine role min bits for {role!r} must be one of {allocate.AFFINE_BITS}")
        out[str(role)] = bit_i
    return dict(sorted(out.items()))


def _validate_budget_split(split: dict[str, float] | None) -> dict[str, float] | None:
    if split is None:
        return None
    if set(split) != {"experts", "affine"}:
        raise ValueError("budget_split must contain exactly 'experts' and 'affine'")
    experts = float(split["experts"])
    affine = float(split["affine"])
    if experts < 0.0 or affine < 0.0:
        raise ValueError("budget_split fractions must be non-negative")
    if abs((experts + affine) - 1.0) > 1e-9:
        raise ValueError("budget_split fractions must sum to 1.0")
    return {"experts": experts, "affine": affine}


def _validate_min_routed_expert_bits(bits: int) -> int:
    bits = int(bits)
    if bits not in allocate.EXPERT_BITS:
        allowed = ", ".join(str(b) for b in allocate.EXPERT_BITS)
        raise ValueError(
            f"min_routed_expert_bits must be one of {{{allowed}}}, got {bits}")
    return bits


def _lossless_mxfp4_capable(unit: dict) -> bool:
    return (
        unit.get("source_codec") == _LOSSLESS_MXFP4_SOURCE_CODEC
        or "mxfp4" in set(unit.get("lossless_codecs", []))
    )


def _expert_min_bits(unit: dict) -> int:
    """Family structural floor for routed experts.

    DeepSeek-V4's first three MoE layers are hash-routed: the route is fixed by
    token id (`tid2eid`) and cannot average quantization noise the way learned
    top-k routing can. Reconstruction-quality proxies underweight that failure
    mode, so source-FP4 experts in those layers start at the lossless 4-bit tier.
    """
    if (
        _lossless_mxfp4_capable(unit)
        and int(unit.get("layer_index", 10**9)) < _DEEPSEEK_V4_HASH_ROUTED_LAYERS
    ):
        return _DEEPSEEK_V4_HASH_EXPERT_MIN_BITS
    return allocate.EXPERT_BITS[0]


def _build_working_sets(
    evidence: dict,
    tau: float | None,
    lm_head_bits: int,
    affine_role_weights: dict[str, float] | None = None,
    affine_role_bit_weights: dict[str, dict[int, float]] | None = None,
    affine_role_min_bits: dict[str, int] | None = None,
    force_tq4_lossless: bool = False,
    force_dense_lossless_mx: bool = False,
    min_routed_expert_bits: int = allocate.EXPERT_BITS[0],
):
    """probe_evidence units -> (experts, affine_tensors, fp16_tensors) working dicts.

    Reads role/layer_index/projection off each unit (typed). No name-parsing.
    """
    role_weights = _validate_affine_role_weights(affine_role_weights)
    role_bit_weights = _validate_affine_role_bit_weights(affine_role_bit_weights)
    role_min_bits = _validate_affine_role_min_bits(affine_role_min_bits)
    experts, affine_tensors, fp16_tensors = [], [], []
    for u in evidence["units"]:
        if u["kind"] == "expert":
            quality = _quality_table_expert(u)
            codec_by_bits = {bits: "tq" for bits in allocate.EXPERT_BITS}
            if _lossless_mxfp4_capable(u) and not force_tq4_lossless:
                quality[4] = 1.0
                codec_by_bits[4] = "mxfp4"
            min_bits = max(_expert_min_bits(u), min_routed_expert_bits)
            experts.append({
                "source_name": u["source_name"],
                "role": u["role"],
                "layer": u["layer_index"],
                "projection": u["projection"],
                "n_experts": u["n_experts"],
                "shape": u["shape"],            # true [out_features, in_features] geometry
                "importance": u.get("importance", 0.0),
                "quality": quality,
                "source_codec": u.get("source_codec"),
                "lossless_mxfp4_capable": _lossless_mxfp4_capable(u),
                "codec_by_bits": codec_by_bits,
                "min_bits": min_bits,
            })
            continue

        rows, cols = u["shape"]
        role = u["role"]
        t = {
            "source_name": u["source_name"],
            "role": role,
            "layer_key": _layer_key(u),
            "rows": rows,
            "cols": cols,
            "importance": u.get("importance", 0.0),
            "quality": _quality_table_affine(u),
            "dense_codec_quality": _quality_table_dense_mx(u),
            "source_codec": u.get("source_codec"),
            "lossless_codecs": list(u.get("lossless_codecs", [])),
        }
        if role in FP16_ROLES:
            fp16_tensors.append(t)
            continue

        if role in role_weights:
            t["objective_importance"] = t["importance"] * role_weights[role]
        if role in role_bit_weights:
            t["objective_bit_weights"] = role_bit_weights[role]

        # Serve-critical safety floors (always on, independent of tau): lm_head and
        # embed_tokens. The backbone gets no per-role floors; the health gate
        # protects it.
        min_bits = _SAFETY_MIN_BITS.get(role, allocate.AFFINE_BITS[0])
        if role == "lm_head":
            min_bits = max(min_bits, lm_head_bits)
        if role in role_min_bits:
            min_bits = max(min_bits, role_min_bits[role])
        t["min_bits"] = min_bits
        if (
            force_dense_lossless_mx
            and "mxfp8" in t["lossless_codecs"]
            and ("mxfp8", 8, 32) in t["dense_codec_quality"]
        ):
            t["min_format"] = "mxfp8"
            t["min_bits"] = 8
            t["min_group_size"] = 32
        affine_tensors.append(t)
    return experts, affine_tensors, fp16_tensors


def _normalize_expert_importance(experts: list[dict]) -> dict[str, float]:
    """Equalize expert importance class means across projections.

    The probe's scalar importance is mean(imatrix second moments) of the
    tensor's input activations: gate/up read the residual stream, down reads
    post-SiLU(gate)*up products ~23x smaller in magnitude. The greedy compares
    these scalars across activation spaces (pri = importance*gain/bytes), so
    raw values starved down_proj to 1-bit at every target: the inverse of
    measured external evidence and the source of the pinned worst-layer tail
    (carried on a 6-question panel at unsloth-equal
    size). Scaling each projection class to the global expert mean keeps the
    within-class layer ranking and makes the cross-class auction fair.
    Returns the applied scales {projection: scale} for the constraints record.
    """
    if not experts:
        return {}
    by_class: dict[str, list[float]] = {}
    for e in experts:
        by_class.setdefault(e["projection"], []).append(e["importance"])
    total = sum(v for vs in by_class.values() for v in vs)
    count = sum(len(vs) for vs in by_class.values())
    if total <= 0:
        return {}
    gmean = total / count
    scales = {}
    for proj, vs in sorted(by_class.items()):
        m = sum(vs) / len(vs)
        scales[proj] = gmean / m if m > 0 else 1.0
    for e in experts:
        e["importance"] *= scales[e["projection"]]
    return scales


def _summarize(experts, affine_tensors, fp16_tensors, expert_bits,
               cur_formats, cur_bits, cur_gs, alpha):
    """Achieved fidelity / tail / size from the chosen allocation."""
    pairs, weighted = [], []      # (layer_key, q) for tail; (importance, q) for F
    expert_size, expert_bit_counts, expert_codec_counts = 0, {}, {}
    for e in experts:
        bits = expert_bits[(e["layer"], e["projection"])]
        codec = allocate.expert_codec(e, bits)
        expert_size += allocate.expert_bytes(e, bits)
        expert_bit_counts[bits] = expert_bit_counts.get(bits, 0) + 1
        expert_codec_counts[codec] = expert_codec_counts.get(codec, 0) + 1
        q = allocate.expert_q(e, bits)
        pairs.append((e["layer"], q))
        weighted.append((e["importance"], q))

    tensor_size, tensor_bit_counts, tensor_format_counts = 0, {}, {}
    for i, t in enumerate(affine_tensors):
        fmt = cur_formats[i]
        if fmt == "affine":
            tensor_size += affine_bytes(t["rows"], t["cols"], cur_bits[i], cur_gs[i])
        else:
            tensor_size += mx_float_bytes(t["rows"], t["cols"], cur_bits[i])
        tensor_bit_counts[cur_bits[i]] = tensor_bit_counts.get(cur_bits[i], 0) + 1
        tensor_format_counts[fmt] = tensor_format_counts.get(fmt, 0) + 1
        q = allocate.dense_q(t, (fmt, cur_bits[i], cur_gs[i]))
        pairs.append((t["layer_key"], q))
        weighted.append((t["importance"], q))

    for t in fp16_tensors:
        tensor_size += t["rows"] * t["cols"] * 2
        pairs.append((t["layer_key"], 1.0))
        weighted.append((t["importance"], 1.0))

    total_size = expert_size + tensor_size
    n_expert_params = sum(e["n_experts"] * e["shape"][0] * e["shape"][1] for e in experts)
    bps = total_size * 8 / n_expert_params if n_expert_params > 0 else 0.0
    return {
        "fidelity": fidelity(weighted),
        "worst_layer_tail": worst_layer_tail(pairs, alpha),
        "size_gb": total_size / (1024 ** 3),
        "expert_size_gb": expert_size / (1024 ** 3),
        "tensor_size_gb": tensor_size / (1024 ** 3),
        "bps": bps,
        "expert_bit_counts": {str(k): v for k, v in sorted(expert_bit_counts.items())},
        "expert_codec_counts": dict(sorted(expert_codec_counts.items())),
        "tensor_format_counts": dict(sorted(tensor_format_counts.items())),
        "tensor_bit_counts": {str(k): v for k, v in sorted(tensor_bit_counts.items())},
    }


def _render_allocation(experts, affine_tensors, fp16_tensors, expert_bits,
                       cur_formats, cur_bits, cur_gs):
    """Per-source allocation list (sorted for a deterministic artifact)."""
    out = []
    for e in experts:
        bits = expert_bits[(e["layer"], e["projection"])]
        codec = allocate.expert_codec(e, bits)
        out.append({
            "source_name": e["source_name"], "kind": "expert", "role": e["role"],
            "layer_index": e["layer"], "projection": e["projection"],
            "bits": bits, "codec": codec,
            "format": "mxfp4" if codec == "mxfp4" else "tq",
            "source_codec": e.get("source_codec"),
            "lossless": bool(codec == "mxfp4" and bits == 4),
        })
    for i, t in enumerate(affine_tensors):
        fmt = cur_formats[i]
        alloc = {
            "source_name": t["source_name"], "kind": "affine", "role": t["role"],
            "bits": cur_bits[i], "group_size": cur_gs[i],
        }
        if fmt != "affine":
            alloc["format"] = fmt
            alloc["source_codec"] = t.get("source_codec")
            alloc["lossless"] = fmt in set(t.get("lossless_codecs", []))
        out.append(alloc)
    for t in fp16_tensors:
        out.append({
            "source_name": t["source_name"], "kind": "fp16_passthrough", "role": t["role"],
        })
    out.sort(key=lambda a: (a.get("layer_index") if a.get("layer_index") is not None else -1,
                            a.get("projection", ""), a["source_name"]))
    return out


def _objective(constraints: dict) -> str:
    """A named, human-readable statement of what the optimizer maximized.

    The optimizer's goal is always 'maximize importance-weighted fidelity F'; the
    constraints it's subject to vary (a worst-layer tail floor, a quality target,
    a size target). We name the actual combination rather than invent a value the
    greedy did not optimize. The spec requires an objective statement rather
    than a fabricated metric.

    On semantics: target_size_gb is a fill-toward target. It does not enforce a
    strict `<=` budget; the greedy upgrades bits per fidelity-per-byte until the package
    reaches (>=) the target (allocate.py: stop when current_size >= target_bytes), so
    it fills up to the size, landing slightly above. And when multiple targets are
    given, the first satisfied stops the greedy (e.g. quality + size: whichever is met
    first wins; a small quality target can stop the run before the size is spent).
    """
    subject_to = []
    if constraints.get("tau") is not None:
        subject_to.append(f"worst_layer_tail(CVaR_{constraints['alpha']}) >= {constraints['tau']}")
    if constraints.get("target_quality") is not None:
        subject_to.append(f"fidelity >= {constraints['target_quality']}")
    if constraints.get("target_size_gb") is not None:
        subject_to.append(f"size_gb ~= {constraints['target_size_gb']} (fill-toward)")
    if constraints.get("min_routed_expert_bits", allocate.EXPERT_BITS[0]) > allocate.EXPERT_BITS[0]:
        subject_to.append(
            f"routed expert bits >= {constraints['min_routed_expert_bits']}")
    if constraints.get("budget_split") is not None:
        split = constraints["budget_split"]
        subject_to.append(
            f"strict expert/affine budget split "
            f"({split['experts']:.0%}/{split['affine']:.0%})")
    goal = "maximize importance-weighted fidelity F per byte"
    if (constraints.get("affine_role_weights")
            or constraints.get("affine_role_bit_weights")
            or constraints.get("affine_role_min_bits")):
        goal = "maximize role-adjusted affine risk reduction per byte"
    return f"{goal} s.t. {', '.join(subject_to)}" if subject_to else goal


def _rejected(experts, affine_tensors, fp16_tensors, constraints: dict) -> list[dict]:
    """Alternatives the optimizer actually rejected, with reasons (spec field).

    Records only real choices the decision made, never a fabricated
    search the greedy didn't run. Genuine rejections:
      - quantizing the router/shared-expert gates (rejected: kept fp16 for
        structural correctness: a reconstruction proxy can't see routing).
    """
    out: list[dict] = []
    if fp16_tensors:
        roles = sorted({t["role"] for t in fp16_tensors})
        out.append({
            "choice": f"quantize {', '.join(roles)}",
            "reason": "kept fp16: router/shared-expert gates steer discrete routing, "
                      "which a reconstruction-error proxy cannot evaluate",
        })
    if any(e.get("lossless_mxfp4_capable") for e in experts):
        if constraints.get("force_tq4_lossless"):
            out.append({
                "choice": "source mxfp4 for lossless-capable routed experts",
                "reason": "disabled by the explicit force_tq4_lossless override",
            })
        else:
            out.append({
                "choice": "TQ4 for lossless-capable routed experts",
                "reason": "source mxfp4 is lossless for the DS4 e2m1 source weights "
                          "and is the default 4-bit routed-expert codec",
            })
    if constraints.get("force_dense_lossless_mx"):
        roles = sorted({
            t["role"] for t in affine_tensors
            if "mxfp8" in t.get("lossless_codecs", [])
        })
        if roles:
            out.append({
                "choice": f"lower dense FP8/e4m3 roles below mxfp8: {', '.join(roles)}",
                "reason": "disabled by --force-dense-lossless-mx: source-compatible "
                          "mxfp8 is the near-lossless dense floor for these tensors",
            })
    if constraints.get("min_routed_expert_bits", allocate.EXPERT_BITS[0]) > allocate.EXPERT_BITS[0]:
        out.append({
            "choice": "TQ1 for routed experts",
            "reason": "disabled by explicit min_routed_expert_bits floor for a "
                      "serve-quality ablation",
        })
    return out


def decide(
    evidence: dict,
    *,
    target_quality: float | None = None,
    target_size_gb: float | None = None,
    tau: float | None = None,
    alpha: float = 0.05,
    lm_head_bits: int = _LM_HEAD_DEFAULT_BITS,
    allow_unhealthy: bool = False,
    affine_role_profile_name: str | None = None,
    affine_role_weights: dict[str, float] | None = None,
    affine_role_bit_weights: dict[str, dict[int | str, float]] | None = None,
    affine_role_min_bits: dict[str, int] | None = None,
    budget_split: dict[str, float] | None = None,
    expert_importance_norm: str | None = "class-mean",
    force_tq4_lossless: bool = False,
    force_dense_lossless_mx: bool = False,
    min_routed_expert_bits: int = allocate.EXPERT_BITS[0],
) -> dict:
    """Allocate bits over a probe_evidence artifact -> optimizer_decision artifact.

    The allocation health gate rejects serve-unviable allocations (collapsed
    backbone / critical tensors) by recording blocking validation + status=invalid, so
    a bad recipe can't silently ship a multi-GB package. `allow_unhealthy=True` is the
    explicit escape hatch: the same findings are still recorded (downgraded to warnings),
    but the decision stays valid.
    """
    required_features = list(evidence.get("required_features", []))
    role_weights = _validate_affine_role_weights(affine_role_weights)
    role_bit_weights = _validate_affine_role_bit_weights(affine_role_bit_weights)
    role_min_bits = _validate_affine_role_min_bits(affine_role_min_bits)
    min_routed_expert_bits = _validate_min_routed_expert_bits(min_routed_expert_bits)
    split = _validate_budget_split(budget_split)
    if split is not None:
        if target_size_gb is None:
            raise ValueError("budget_split requires target_size_gb")
        if target_quality is not None:
            raise ValueError("budget_split does not support target_quality")
        if tau is not None:
            raise ValueError("budget_split does not support tau")
    if expert_importance_norm not in ("class-mean", "off", None):
        raise ValueError(
            f"expert_importance_norm must be 'class-mean', 'off' or None, "
            f"got {expert_importance_norm!r}")
    experts, affine_tensors, fp16_tensors = _build_working_sets(
        evidence,
        tau,
        lm_head_bits,
        role_weights,
        role_bit_weights,
        role_min_bits,
        force_tq4_lossless=bool(force_tq4_lossless),
        force_dense_lossless_mx=bool(force_dense_lossless_mx),
        min_routed_expert_bits=min_routed_expert_bits,
    )
    importance_scales = {}
    if expert_importance_norm == "class-mean":
        importance_scales = _normalize_expert_importance(experts)

    constraints = {"target_quality": target_quality, "target_size_gb": target_size_gb,
                   "tau": tau, "alpha": alpha, "lm_head_bits": lm_head_bits,
                   "force_tq4_lossless": bool(force_tq4_lossless),
                   "force_dense_lossless_mx": bool(force_dense_lossless_mx),
                   "min_routed_expert_bits": min_routed_expert_bits}
    if importance_scales:
        constraints["expert_importance_norm"] = {
            "mode": "class-mean",
            "scales": {p: round(s, 6) for p, s in importance_scales.items()},
        }
    if split is not None:
        constraints["budget_split"] = split
    if affine_role_profile_name is not None:
        if not affine_role_profile_name:
            raise ValueError("affine_role_profile_name must be non-empty")
        constraints["affine_role_profile_name"] = affine_role_profile_name
    if role_weights:
        constraints["affine_role_weights"] = role_weights
    if role_bit_weights:
        constraints["affine_role_bit_weights"] = {
            role: {str(bit): weight for bit, weight in bit_map.items()}
            for role, bit_map in role_bit_weights.items()
        }
    if role_min_bits:
        constraints["affine_role_min_bits"] = role_min_bits
    objective = _objective(constraints)
    rejected = _rejected(experts, affine_tensors, fp16_tensors, constraints)

    feasibility = "feasible"
    # Report (don't hide) genuine q-table inversions; non-blocking; the envelope
    # already smoothed them for the allocation.
    validation: list[Validation] = _q_inversion_validations(evidence)
    try:
        result = allocate.optimize(
            experts, affine_tensors, fp16_tensors,
            target_quality=target_quality, target_size_gb=target_size_gb,
            tau=tau, alpha=alpha, budget_split=split)
    except allocate.InfeasibleUnderBudget as exc:
        feasibility = "infeasible_under_budget"
        validation.append(Validation("error", "optimize.infeasible_under_budget",
                                     str(exc), phase="optimize", blocking=True))
        result = None
    except allocate.Infeasible as exc:
        feasibility = "infeasible"
        validation.append(Validation("error", "optimize.infeasible",
                                     str(exc), phase="optimize", blocking=True))
        result = None

    if result is None:
        return make_artifact(
            "optimizer_decision", evidence["subject"], PRODUCER,
            required_features=required_features,
            status="invalid", validation=validation,
            objective=objective, constraints=constraints, feasibility=feasibility,
            allocation=[], achieved=None, rejected=rejected,
            source_probe_id=evidence.get("artifact_id"),
        )

    expert_bits = result["expert_bits"]
    cur_formats = result.get("cur_formats", ["affine" for _ in result["cur_bits"]])
    cur_bits, cur_gs = result["cur_bits"], result["cur_gs"]
    allocation = _render_allocation(experts, affine_tensors, fp16_tensors,
                                    expert_bits, cur_formats, cur_bits, cur_gs)
    achieved = _summarize(experts, affine_tensors, fp16_tensors,
                          expert_bits, cur_formats, cur_bits, cur_gs, alpha)
    if result.get("budget_split") is not None:
        achieved["budget_split"] = result["budget_split"]

    # Allocation health gate: reject serve-unviable allocations the proxy can't
    # see. allow_unhealthy downgrades the findings to warnings (recorded, non-blocking).
    health = health_check(allocation)
    status = "valid"
    if health:
        if allow_unhealthy:
            for v in health:
                validation.append(Validation("warning", v.code, v.message, path=v.path,
                                              phase=v.phase, blocking=False,
                                              expected=v.expected, actual=v.actual))
            validation.append(Validation(
                "info", "optimize.health_overridden",
                "allocation failed the health gate but was accepted via allow_unhealthy",
                phase="optimize", blocking=False))
        else:
            validation.extend(health)
            status = "invalid"

    return make_artifact(
        "optimizer_decision", evidence["subject"], PRODUCER,
        required_features=required_features,
        status=status, validation=validation,
        objective=objective, constraints=constraints, feasibility=feasibility,
        allocation=allocation, achieved=achieved, rejected=rejected,
        source_probe_id=evidence.get("artifact_id"),
    )
