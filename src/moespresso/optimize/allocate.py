"""The bit-allocation core: feasibility-first tail constraint, then fidelity greedy.

This module is format- and name-neutral: it operates on plain working dicts
(expert / affine) carrying layer_key, importance, shape, and a quality table.
The adapter in `decide.py` builds those dicts from a probe_evidence artifact's
typed roles; name-parsing never happens here.

Two phases:
  1. _satisfy_tail: lift bits until CVaR_alpha over per-layer minima >= tau, so a
     single catastrophic layer can't hide behind a good mean. Bundles tied minima
     (lifting one argmin leaves the layer min unchanged) and recomputes the
     non-separable delta-tail each step.
  2. fidelity greedy: maximize importance-weighted F per byte via upgrades only,
     which preserves the tail feasibility established in phase 1.
"""

from __future__ import annotations

import heapq
import math

from moespresso.optimize.aggregate import layer_minima, worst_layer_tail
from moespresso.optimize.sizes import (
    affine_bytes,
    mxfp4_expert_bytes,
    mx_float_bytes,
    tq_expert_bytes,
)

EXPERT_BITS = [1, 2, 4]
AFFINE_BITS = [2, 3, 4, 5, 6, 8]
GROUP_SIZES = [128, 64, 32]
DENSE_MX_FORMATS = {"mxfp4", "mxfp8"}


class Infeasible(Exception):
    """The worst-layer tail floor tau cannot be met even at max bits."""


class InfeasibleUnderBudget(Exception):
    """The worst-layer tail floor tau cannot be met within the size budget."""


# --- quality lookups on the working dicts ---

def expert_q(e: dict, bits: int) -> float:
    return e["quality"].get(bits, 0.0)


def expert_codec(e: dict, bits: int) -> str:
    return e.get("codec_by_bits", {}).get(bits, "tq")


def expert_bytes(e: dict, bits: int) -> int:
    r, c = e["shape"]
    if expert_codec(e, bits) == "mxfp4":
        return mxfp4_expert_bytes(e["n_experts"], r, c)
    return tq_expert_bytes(e["n_experts"], r, c, bits)


def _dense_option(format_: str, bits: int, gs: int) -> tuple[str, int, int]:
    return (format_, int(bits), int(gs))


def _iter_dense_quality(t: dict):
    """Yield ((format, bits, group_size), q) for legacy affine and MX tables."""
    for key, value in t.get("quality", {}).items():
        if isinstance(key, tuple):
            if len(key) == 2:
                bits, gs = key
                yield _dense_option("affine", bits, gs), value
            elif len(key) == 3:
                fmt, bits, gs = key
                yield _dense_option(fmt, bits, gs), value
            continue
        parts = str(key).split("_")
        if len(parts) == 2:
            bits, gs = parts
            yield _dense_option("affine", bits, gs), value
    for key, value in t.get("dense_codec_quality", {}).items():
        if isinstance(key, tuple) and len(key) == 3:
            fmt, bits, gs = key
            if fmt in DENSE_MX_FORMATS:
                yield _dense_option(fmt, bits, gs), value
            continue
        parts = str(key).split("_")
        if len(parts) == 3 and parts[0] in DENSE_MX_FORMATS:
            fmt, bits, gs = parts
            yield _dense_option(fmt, bits, gs), value


def _dense_quality_map(t: dict) -> dict[tuple[str, int, int], float]:
    return dict(_iter_dense_quality(t))


def dense_q(t: dict, option: tuple[str, int, int]) -> float:
    return _dense_quality_map(t).get(option, 0.0)


def affine_q(t: dict, bits: int, gs: int) -> float:
    return dense_q(t, _dense_option("affine", bits, gs))


def dense_bytes(t: dict, option: tuple[str, int, int]) -> int:
    fmt, bits, gs = option
    if fmt == "affine":
        return affine_bytes(t["rows"], t["cols"], bits, gs)
    if fmt in DENSE_MX_FORMATS:
        return mx_float_bytes(t["rows"], t["cols"], bits)
    raise ValueError(f"unsupported dense codec {fmt!r}")


def affine_best_q(t: dict) -> float:
    """Best measured quality for a tensor over all (bits, gs) tuples.

    Used by the tail feasibility check: a tensor's reachable ceiling spans every
    measured tuple. Restricting it to max_bits at the current group_size could
    wrongly declare a tau infeasible.
    """
    values = [q for _opt, q in _iter_dense_quality(t)]
    return max(values) if values else 0.0


# --- worst-layer feasibility ---

def _all_units(experts, expert_bits, affine_tensors, cur_formats, cur_bits, cur_gs):
    units = []
    for ei, e in enumerate(experts):
        q = expert_q(e, expert_bits[(e["layer"], e["projection"])])
        units.append((e["layer"], q, ("expert", ei)))
    for ti, t in enumerate(affine_tensors):
        opt = _dense_option(cur_formats[ti], cur_bits[ti], cur_gs[ti])
        units.append((t["layer_key"], dense_q(t, opt), ("affine", ti)))
    return units


def _lift_option(ref, experts, expert_bits, affine_tensors, cur_formats, cur_bits, cur_gs):
    """One tail-lift for a unit: (ref, new_fmt, new_bits, new_gs, new_q, size_cost) or None.

    Expert: next bit up (gs is N/A, returned as None). Affine: the frontier move with the
    best quality-gain-per-byte, including diagonal (bits,gs) moves. This lets the tail
    choose a cheaper or finer group size in addition to the next bit.
    """
    kind, idx = ref
    if kind == "expert":
        e = experts[idx]
        cur = expert_bits[(e["layer"], e["projection"])]
        if cur not in EXPERT_BITS or EXPERT_BITS.index(cur) + 1 >= len(EXPERT_BITS):
            return None
        nb = EXPERT_BITS[EXPERT_BITS.index(cur) + 1]
        cost = expert_bytes(e, nb) - expert_bytes(e, cur)
        return (ref, None, nb, None, expert_q(e, nb), cost)
    t = affine_tensors[idx]
    moves = dense_frontier_moves(t, cur_formats[idx], cur_bits[idx], cur_gs[idx])
    if not moves:
        return None
    fmt_to, b_to, gs_to, q_gain, size_cost = max(moves, key=lambda m: m[3] / m[4])
    return (ref, fmt_to, b_to, gs_to, dense_q(t, _dense_option(fmt_to, b_to, gs_to)), size_cost)


def _apply_lift(ref, new_format, new_bits, new_gs, experts, expert_bits,
                cur_formats, cur_bits, cur_gs):
    kind, idx = ref
    if kind == "expert":
        e = experts[idx]
        expert_bits[(e["layer"], e["projection"])] = new_bits
    else:
        cur_formats[idx] = new_format
        cur_bits[idx] = new_bits
        if new_gs is not None:
            cur_gs[idx] = new_gs


def _satisfy_tail(experts, expert_bits, affine_tensors, cur_formats, cur_bits, cur_gs,
                  tau, alpha):
    """Lift bits until CVaR_alpha over per-layer minima >= tau. Raises Infeasible."""
    eps = 1e-9

    # Reachable ceiling per unit uses the best q over all tuples (affine: any bits×gs).
    # Restricting the ceiling to the current group size can create false infeasibility.
    max_pairs = [(e["layer"], expert_q(e, EXPERT_BITS[-1])) for e in experts]
    for t in affine_tensors:
        max_pairs.append((t["layer_key"], affine_best_q(t)))
    best_possible = worst_layer_tail(max_pairs, alpha)
    if best_possible < tau - eps:
        raise Infeasible(
            f"tail floor tau={tau} unreachable; best achievable tail={best_possible:.4f}")

    while True:
        units = _all_units(experts, expert_bits, affine_tensors, cur_formats, cur_bits, cur_gs)
        pairs = [(lk, q) for lk, q, _ in units]
        cur_tail = worst_layer_tail(pairs, alpha)
        if cur_tail >= tau - eps:
            return
        lm = layer_minima(pairs)
        k = max(1, math.ceil(alpha * len(lm)))
        tail_layers = sorted(lm, key=lm.get)[:k]

        best = None  # (delta_per_byte, [lift_option, ...])
        for tl in tail_layers:
            mval = lm[tl]
            lifts = []
            for lk, q, ref in units:
                if lk == tl and q <= mval + eps:
                    opt = _lift_option(ref, experts, expert_bits, affine_tensors,
                                       cur_formats, cur_bits, cur_gs)
                    if opt is not None:
                        lifts.append(opt)
            if not lifts:
                continue
            lifted = {o[0]: o[4] for o in lifts}   # ref -> new_q
            new_pairs = [(lk, lifted.get(ref, q)) for lk, q, ref in units]
            delta = worst_layer_tail(new_pairs, alpha) - cur_tail
            cost = sum(o[5] for o in lifts)        # size_cost
            if delta > eps and cost > 0:
                ppb = delta / cost
                if best is None or ppb > best[0]:
                    best = (ppb, lifts)

        if best is None:
            raise Infeasible(
                f"tail floor tau={tau} unreachable; stuck at tail={cur_tail:.4f} "
                f"(no productive tail-layer upgrade)")
        for ref, nfmt, nb, ngs, _nq, _cost in best[1]:
            _apply_lift(ref, nfmt, nb, ngs, experts, expert_bits,
                        cur_formats, cur_bits, cur_gs)


# --- fidelity-greedy upgrade pushes ---

def _push_expert_upgrade(heap, ei, e, cur_b):
    if cur_b not in EXPERT_BITS or EXPERT_BITS.index(cur_b) + 1 >= len(EXPERT_BITS):
        return
    next_b = EXPERT_BITS[EXPERT_BITS.index(cur_b) + 1]
    q_gain = expert_q(e, next_b) - expert_q(e, cur_b)
    size_cost = expert_bytes(e, next_b) - expert_bytes(e, cur_b)
    if q_gain > 0 and size_cost > 0:
        pri = e["importance"] * q_gain / size_cost
        heapq.heappush(heap, (-pri, "expert", ei, "expert", cur_b, 0,
                              "expert", next_b, 0))


def _objective_importance(t: dict) -> float:
    """Importance used by the affine allocation objective.

    Existing working dicts only carry `importance`, so this is behavior-neutral.
    Role-elasticity calibration can set `objective_importance` to change upgrade
    pressure without mutating the original probe evidence or reported metrics.
    """
    return t.get("objective_importance", t["importance"])


def _move_objective_importance(t: dict, bits_to: int) -> float:
    """Move priority weight for an affine upgrade.

    `objective_importance` changes the tensor's general objective pressure.
    `objective_bit_weights` is narrower: it biases which destination bit-width is
    worth buying for that role. It is priority-only, meant for target-size role
    band experiments; reported fidelity still comes from the original q-table.
    """
    bit_weights = t.get("objective_bit_weights")
    if not bit_weights:
        return _objective_importance(t)
    return _objective_importance(t) * float(bit_weights.get(bits_to, 1.0))


def dense_frontier_moves(t, cur_format, cur_bits, cur_gs):
    """Improving, non-dominated dense-codec moves from the current option.

    Returns [(format_to, bits_to, group_size_to, q_gain, size_cost), ...].
    Legacy affine tuples and new MX float tuples share the same local Pareto
    frontier, so the optimizer can choose `mxfp8` only when measured evidence
    justifies buying its bytes.
    """
    cur_opt = _dense_option(cur_format, cur_bits, cur_gs)
    cur_q = dense_q(t, cur_opt)
    cur_size = dense_bytes(t, cur_opt)
    cand = []  # (fmt, bits, gs, q, size)
    for opt, q in _iter_dense_quality(t):
        size = dense_bytes(t, opt)
        if q - cur_q > 0 and size - cur_size > 0:
            cand.append((*opt, q, size))
    frontier = []
    for (fmt, b, gs, q, size) in cand:
        dominated = any(
            (oq >= q and osize <= size) and (oq > q or osize < size)
            for (ofmt, ob, ogs, oq, osize) in cand
            if (ofmt, ob, ogs) != (fmt, b, gs)
        )
        if not dominated:
            frontier.append((fmt, b, gs, q - cur_q, size - cur_size))
    return frontier


def affine_frontier_moves(t, cur_bits, cur_gs):
    """Improving, non-dominated (bits, group_size) moves from the current tuple.

    Enumerates all measured (bits, gs) for the tensor (≤ |AFFINE_BITS|×|GROUP_SIZES| = 18)
    (not just axis-aligned next-steps) so diagonal moves like (4,128)->(6,32) are
    reachable in one hop (the old search couldn't see them). Keeps a target only if it
    buys quality (q_gain > 0) at positive size_cost, then drops dominated targets: a
    candidate beaten by another with q >= it and size <= it (one strict) is never offered.
    Returns [(b_to, gs_to, q_gain, size_cost), ...]. This is a local per-tensor
    frontier; a global risk[t,bits,gs] objective can reuse the same candidate set.
    """
    return [
        (b, gs, q_gain, size_cost)
        for fmt, b, gs, q_gain, size_cost
        in dense_frontier_moves(t, "affine", cur_bits, cur_gs)
        if fmt == "affine"
    ]


def _push_affine_upgrades(heap, i, t, cur_format, cur_bits, cur_gs):
    """Push the single best-value-per-byte frontier move (one heap entry per tensor).

    The frontier (affine_frontier_moves) includes diagonal (bits,gs) moves, so the best
    move can be diagonal. Only the best move is pushed because adding the whole frontier's
    ~18 candidates and re-pushing after every accepted hop blew up the heap (O(steps ×
    frontier) → seconds per optimize on a 35B). After this move is taken, the greedy
    re-pushes from the new state, so the rest of the frontier is still explored over time.
    """
    moves = dense_frontier_moves(t, cur_format, cur_bits, cur_gs)
    if not moves:
        return
    fmt_to, b_to, gs_to, q_gain, size_cost = max(
        moves,
        key=lambda m: _move_objective_importance(t, m[1]) * m[3] / m[4],
    )
    pri = _move_objective_importance(t, b_to) * q_gain / size_cost
    heapq.heappush(heap, (-pri, "affine", i,
                          cur_format, cur_bits, cur_gs, fmt_to, b_to, gs_to))


# --- optimization ---

def optimize(
    experts: list[dict],
    affine_tensors: list[dict],
    fp16_tensors: list[dict],
    *,
    target_quality: float | None = None,
    target_size_gb: float | None = None,
    tau: float | None = None,
    alpha: float = 0.05,
    budget_split: dict[str, float] | None = None,
) -> dict:
    """Allocate bits. Returns {expert_bits, cur_bits, cur_gs} for the caller to render.

    `experts`/`affine_tensors`/`fp16_tensors` are the working dicts built by the
    adapter. min_bits floors come pre-set on the affine dicts as t["min_bits"].
    """
    if target_quality is None and target_size_gb is None:
        raise ValueError("Provide target_quality or target_size_gb")

    fp16_size = sum(t["rows"] * t["cols"] * 2 for t in fp16_tensors)
    fp16_imp = sum(t["importance"] for t in fp16_tensors)

    # Start every expert/affine tensor at its min_bits floor. The adapter sets
    # min_bits = the lowest supported tier for almost everything; family-specific
    # structural floors (for example DS4 hash-routed source-FP4 experts) carry a
    # higher value. Greedy and _satisfy_tail only ever lift bits, so starting at
    # the floor preserves it while objective/tail upgrades can add more on top.
    expert_bits = {
        (e["layer"], e["projection"]): e.get("min_bits", EXPERT_BITS[0])
        for e in experts
    }
    cur_formats = [t.get("min_format", "affine") for t in affine_tensors]
    cur_bits = [t.get("min_bits", AFFINE_BITS[0]) for t in affine_tensors]
    cur_gs = [t.get("min_group_size", GROUP_SIZES[0]) for t in affine_tensors]

    if tau is not None:
        _satisfy_tail(
            experts, expert_bits, affine_tensors, cur_formats, cur_bits, cur_gs,
            tau, alpha)

    total_imp = (sum(e["importance"] for e in experts)
                 + sum(_objective_importance(t) for t in affine_tensors) + fp16_imp)

    def _state_size():
        s = fp16_size
        for e in experts:
            s += expert_bytes(e, expert_bits[(e["layer"], e["projection"])])
        for i, t in enumerate(affine_tensors):
            s += dense_bytes(t, _dense_option(cur_formats[i], cur_bits[i], cur_gs[i]))
        return s

    def _state_loss():
        loss = 0.0
        for e in experts:
            loss += e["importance"] * (1 - expert_q(e, expert_bits[(e["layer"], e["projection"])]))
        for i, t in enumerate(affine_tensors):
            opt = _dense_option(cur_formats[i], cur_bits[i], cur_gs[i])
            loss += _objective_importance(t) * (1 - dense_q(t, opt))
        return loss

    current_size = _state_size()
    current_loss = _state_loss()

    target_bytes = target_size_gb * (1024 ** 3) if target_size_gb is not None else None
    if target_bytes is not None and tau is not None and current_size > target_bytes:
        raise InfeasibleUnderBudget(
            f"tail floor tau={tau} needs {current_size / 1024 ** 3:.2f} GB "
            f"> budget {target_size_gb} GB")

    def _quality():
        return 1 - current_loss / total_imp if total_imp > 0 else 1.0

    def _result():
        return {
            "expert_bits": expert_bits,
            "cur_formats": cur_formats,
            "cur_bits": cur_bits,
            "cur_gs": cur_gs,
        }

    if budget_split is not None:
        expert_base = sum(
            expert_bytes(e, expert_bits[(e["layer"], e["projection"])])
            for e in experts
        )
        affine_base = sum(
            dense_bytes(t, _dense_option(cur_formats[i], cur_bits[i], cur_gs[i]))
            for i, t in enumerate(affine_tensors)
        )
        remaining = max(0.0, float(target_bytes or 0) - fp16_size - expert_base - affine_base)
        expert_budget = remaining * budget_split["experts"]
        affine_budget = remaining * budget_split["affine"]
        split_stats = {
            "expert_budget_bytes": int(expert_budget),
            "affine_budget_bytes": int(affine_budget),
            "expert_spent_bytes": 0,
            "affine_spent_bytes": 0,
            "expert_unused_bytes": int(expert_budget),
            "affine_unused_bytes": int(affine_budget),
        }

        expert_heap: list[tuple] = []
        for ei, e in enumerate(experts):
            _push_expert_upgrade(expert_heap, ei, e, expert_bits[(e["layer"], e["projection"])])
        while expert_heap:
            (_neg_pri, _kind, idx, _fmt_from, b_from, _gs_from,
             _fmt_to, b_to, _gs_to) = heapq.heappop(expert_heap)
            e = experts[idx]
            key = (e["layer"], e["projection"])
            if expert_bits[key] != b_from:
                _push_expert_upgrade(expert_heap, idx, e, expert_bits[key])
                continue
            q_gain = expert_q(e, b_to) - expert_q(e, b_from)
            size_cost = expert_bytes(e, b_to) - expert_bytes(e, b_from)
            if q_gain <= 0 or size_cost <= 0:
                continue
            if split_stats["expert_spent_bytes"] + size_cost > expert_budget:
                continue
            expert_bits[key] = b_to
            split_stats["expert_spent_bytes"] += size_cost
            split_stats["expert_unused_bytes"] = int(
                expert_budget - split_stats["expert_spent_bytes"])
            _push_expert_upgrade(expert_heap, idx, e, b_to)

        affine_heap: list[tuple] = []
        for ti, t in enumerate(affine_tensors):
            _push_affine_upgrades(
                affine_heap, ti, t, cur_formats[ti], cur_bits[ti], cur_gs[ti])
        while affine_heap:
            (_neg_pri, _kind, idx, fmt_from, b_from, gs_from,
             fmt_to, b_to, gs_to) = heapq.heappop(affine_heap)
            t = affine_tensors[idx]
            if (
                cur_formats[idx] != fmt_from
                or cur_bits[idx] != b_from
                or cur_gs[idx] != gs_from
            ):
                _push_affine_upgrades(
                    affine_heap, idx, t, cur_formats[idx], cur_bits[idx], cur_gs[idx])
                continue
            opt_from = _dense_option(fmt_from, b_from, gs_from)
            opt_to = _dense_option(fmt_to, b_to, gs_to)
            q_gain = dense_q(t, opt_to) - dense_q(t, opt_from)
            size_cost = dense_bytes(t, opt_to) - dense_bytes(t, opt_from)
            if q_gain <= 0 or size_cost <= 0:
                continue
            if split_stats["affine_spent_bytes"] + size_cost > affine_budget:
                continue
            cur_formats[idx] = fmt_to
            cur_bits[idx] = b_to
            cur_gs[idx] = gs_to
            split_stats["affine_spent_bytes"] += size_cost
            split_stats["affine_unused_bytes"] = int(
                affine_budget - split_stats["affine_spent_bytes"])
            _push_affine_upgrades(affine_heap, idx, t, fmt_to, b_to, gs_to)

        result = _result()
        result["budget_split"] = split_stats
        return result

    if target_quality is not None and _quality() >= target_quality:
        return _result()
    if target_bytes is not None and current_size >= target_bytes:
        return _result()

    heap: list[tuple] = []
    for ei, e in enumerate(experts):
        _push_expert_upgrade(heap, ei, e, expert_bits[(e["layer"], e["projection"])])
    for ti, t in enumerate(affine_tensors):
        _push_affine_upgrades(heap, ti, t, cur_formats[ti], cur_bits[ti], cur_gs[ti])

    while heap:
        item = heapq.heappop(heap)
        _neg_pri, kind, idx = item[:3]
        if kind == "expert":
            (_neg_pri, _kind, idx, _fmt_from, b_from, _gs_from,
             _fmt_to, b_to, _gs_to) = item
            e = experts[idx]
            key = (e["layer"], e["projection"])
            if expert_bits[key] != b_from:
                _push_expert_upgrade(heap, idx, e, expert_bits[key])
                continue
            q_gain = expert_q(e, b_to) - expert_q(e, b_from)
            size_cost = expert_bytes(e, b_to) - expert_bytes(e, b_from)
            if q_gain <= 0 or size_cost <= 0:
                continue
            expert_bits[key] = b_to
            current_size += size_cost
            current_loss -= e["importance"] * q_gain
            _push_expert_upgrade(heap, idx, e, b_to)
        else:
            (_neg_pri, _kind, idx, fmt_from, b_from, gs_from,
             fmt_to, b_to, gs_to) = item
            t = affine_tensors[idx]
            if (
                cur_formats[idx] != fmt_from
                or cur_bits[idx] != b_from
                or cur_gs[idx] != gs_from
            ):
                _push_affine_upgrades(
                    heap, idx, t, cur_formats[idx], cur_bits[idx], cur_gs[idx])
                continue
            opt_from = _dense_option(fmt_from, b_from, gs_from)
            opt_to = _dense_option(fmt_to, b_to, gs_to)
            q_gain = dense_q(t, opt_to) - dense_q(t, opt_from)
            size_cost = dense_bytes(t, opt_to) - dense_bytes(t, opt_from)
            if q_gain <= 0 or size_cost <= 0:
                continue
            cur_formats[idx] = fmt_to
            cur_bits[idx] = b_to
            cur_gs[idx] = gs_to
            current_size += size_cost
            current_loss -= _objective_importance(t) * q_gain
            _push_affine_upgrades(heap, idx, t, fmt_to, b_to, gs_to)

        if target_quality is not None and _quality() >= target_quality:
            return _result()
        if target_bytes is not None and current_size >= target_bytes:
            return _result()

    return _result()
