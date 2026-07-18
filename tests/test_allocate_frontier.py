"""Affine (bits, group_size) tuple search: Pareto-frontier candidate generation.

An axis-aligned-only search (bits up at same gs, or gs up at same bits) with a
tail that lifts bits only can miss the true best tuple: a tensor whose optimum is e.g.
(6b, gs32) is unreachable if neither single-axis step pays off alone but the diagonal
does. `affine_frontier_moves` enumerates all measured (bits,gs), keeps the improving
non-dominated ones (incl. diagonals), so the greedy can take (4,128)->(6,32) in one hop.
Local frontier only; not a global knapsack/ILP.
"""

from __future__ import annotations

from moespresso.optimize.allocate import affine_frontier_moves, optimize
from moespresso.optimize.sizes import affine_bytes


def _tensor(quality, rows=64, cols=128, importance=1.0):
    return {"rows": rows, "cols": cols, "importance": importance, "quality": quality}


def test_frontier_includes_a_diagonal_move():
    # From (2,128): (4,128) and (2,64) each give only a small q bump, but (6,32) is a big
    # jump for its cost. The diagonal must be offered as a candidate.
    q = {(2, 128): 0.80, (4, 128): 0.82, (2, 64): 0.81, (6, 32): 0.99}
    moves = affine_frontier_moves(_tensor(q), 2, 128)
    targets = {(b, gs) for (b, gs, _g, _c) in moves}
    assert (6, 32) in targets            # the diagonal is reachable in one hop


def test_frontier_drops_dominated_targets():
    # (4,64) here has q <= (4,128) but costs more (finer gs = more groups) -> dominated,
    # must not be offered. (Same bits, finer gs, no quality gain = strictly worse.)
    q = {(2, 128): 0.80, (4, 128): 0.95, (4, 64): 0.95}
    moves = affine_frontier_moves(_tensor(q), 2, 128)
    targets = {(b, gs) for (b, gs, _g, _c) in moves}
    assert (4, 128) in targets
    assert (4, 64) not in targets        # dominated by (4,128): equal q, larger size


def test_frontier_only_improving_moves():
    # A target with q <= current is never a candidate (no quality to buy).
    q = {(4, 128): 0.90, (2, 128): 0.70, (6, 128): 0.88}
    moves = affine_frontier_moves(_tensor(q), 4, 128)
    targets = {(b, gs) for (b, gs, _g, _c) in moves}
    assert (2, 128) not in targets       # lower q
    assert (6, 128) not in targets       # 6b measured below current 4b (q_gain<=0)


def test_frontier_candidates_carry_qgain_and_poscost():
    q = {(2, 128): 0.80, (6, 32): 0.99}
    moves = affine_frontier_moves(_tensor(q), 2, 128)
    assert moves
    for (b, gs, q_gain, size_cost) in moves:
        assert q_gain > 0 and size_cost > 0


def test_frontier_empty_at_top_choice():
    # Already at the best measured tuple -> nothing improving to offer.
    q = {(2, 128): 0.80, (8, 32): 0.99}
    assert affine_frontier_moves(_tensor(q), 8, 32) == []


def test_default_importance_preserves_affine_upgrade_order():
    # Equal moves fall back to deterministic tensor order when no role-risk
    # objective weight is present. This pins the behavior-neutral default.
    q = {(2, 128): 0.80, (4, 128): 0.90}
    tensors = [
        {**_tensor(q), "layer_key": 0, "min_bits": 2},
        {**_tensor(q), "layer_key": 1, "min_bits": 2},
    ]
    start = 2 * affine_bytes(64, 128, 2, 128)
    one_upgrade = affine_bytes(64, 128, 4, 128) - affine_bytes(64, 128, 2, 128)

    result = optimize([], tensors, [], target_size_gb=(start + one_upgrade) / (1024 ** 3))

    assert result["cur_bits"] == [4, 2]


def test_objective_importance_changes_affine_upgrade_order():
    # Role-risk calibration can raise objective_importance for a fragile tensor.
    # With equal local q-gains and sizes, the higher objective weight wins the one
    # available upgrade even though that tensor appears second.
    q = {(2, 128): 0.80, (4, 128): 0.90}
    tensors = [
        {**_tensor(q), "layer_key": 0, "min_bits": 2},
        {**_tensor(q), "layer_key": 1, "min_bits": 2, "objective_importance": 10.0},
    ]
    start = 2 * affine_bytes(64, 128, 2, 128)
    one_upgrade = affine_bytes(64, 128, 4, 128) - affine_bytes(64, 128, 2, 128)

    result = optimize([], tensors, [], target_size_gb=(start + one_upgrade) / (1024 ** 3))

    assert result["cur_bits"] == [2, 4]


def test_objective_bit_weights_can_prefer_a_role_band_destination():
    # By local q-gain-per-byte, this tensor would jump straight to 8b. A bit-prior
    # can make the 4b destination the best move instead, which is the mechanism
    # needed for role-coherent bands without forcing a hard cap.
    q = {(2, 128): 0.80, (4, 128): 0.81, (8, 128): 0.95}
    tensor = {
        **_tensor(q),
        "layer_key": 0,
        "min_bits": 2,
        "objective_bit_weights": {4: 100.0, 8: 0.01},
    }
    start = affine_bytes(64, 128, 2, 128)
    one_upgrade = affine_bytes(64, 128, 4, 128) - affine_bytes(64, 128, 2, 128)

    result = optimize([], [tensor], [], target_size_gb=(start + one_upgrade) / (1024 ** 3))

    assert result["cur_bits"] == [4]


def test_dense_mx_option_can_win_when_probe_measured_it():
    q = {(2, 128): 0.20, (8, 128): 0.30}
    tensor = {
        **_tensor(q),
        "layer_key": 0,
        "min_bits": 2,
        "dense_codec_quality": {"mxfp8_8_32": 0.99},
    }

    result = optimize([], [tensor], [], target_quality=0.95)

    assert result["cur_formats"] == ["mxfp8"]
    assert result["cur_bits"] == [8]
    assert result["cur_gs"] == [32]


def test_optimize_scales_at_real_model_size():
    # Guard against the frontier-search blow-up: pushing the entire frontier (~18 moves)
    # per state and re-pushing after every hop made optimize() take ~7s on a 35B-sized
    # input (and hang on real evidence). We push only the single best-per-byte move, so
    # this must stay well under a second at 392 affine + 120 expert units.
    import time
    from moespresso.optimize.allocate import optimize, AFFINE_BITS, GROUP_SIZES

    def aff(i):
        q = {(b, gs): min(0.999, 0.80 + 0.03 * bi + 0.005 * gi)
             for bi, b in enumerate(AFFINE_BITS) for gi, gs in enumerate(GROUP_SIZES)}
        return {"layer_key": i % 40, "rows": 2048, "cols": 2048, "importance": 1.0,
                "min_bits": 2, "quality": q}

    def exp(i):
        return {"layer": i % 40, "projection": "gate", "n_experts": 256,
                "shape": [512, 2048], "importance": 1.0,
                "quality": {1: 0.55, 2: 0.87, 4: 0.99}}

    affine = [aff(i) for i in range(392)]
    experts = [exp(i) for i in range(120)]
    t0 = time.time()
    optimize(experts, affine, [], target_size_gb=17.0)
    assert time.time() - t0 < 2.0      # generous; real run is ~0.04s
