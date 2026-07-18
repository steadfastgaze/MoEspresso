# Optimizer: `optimizer_decision` from `probe_evidence`

`src/moespresso/optimize/` turns a measured `probe_evidence` artifact plus a
constraint set into a per-tensor / per-expert bit allocation, recorded as a
durable, content-addressed `optimizer_decision` artifact. It reads structure from
the typed roles, never from tensor names, and the output is a hashed artifact
rather than a report.

The optimizer's goal is always the same: maximize importance-weighted fidelity
F per byte. What varies is the constraint set it does that subject to: a
worst-layer tail floor, a quality target, a fill-toward size target, or an
expert/affine budget split.

## Module map

| module | responsibility |
|---|---|
| `decide.py` | adapter (typed `probe_evidence` units → working dicts) + artifact emitter; owns the safety floors, the objective string, the health-gate wiring, the q-inversion warnings, and infeasibility-as-artifact |
| `allocate.py` | the bit-allocation core: feasibility-first tail constraint, then the importance-weighted fidelity-per-byte greedy. Format- and name-neutral: operates only on plain working dicts |
| `aggregate.py` | pure proxy scalars: fidelity F (importance-weighted mean q) and the worst-layer tail T_α (CVaR over per-layer minima) |
| `health.py` | the allocation health gate: a pure, model-free check that rejects serve-unviable allocations |
| `monotone.py` | the monotone-in-bits quality envelope that smooths sampling-noise inversions before allocation, and reports genuine ones |
| `sizes.py` | packed-size byte formulas for the quantized formats (affine MLX, TQ expert, source-mxfp4 expert, dense MX float) |
| `affine_elasticity.py` | the calibrated affine role-elasticity profiles (role weights, destination-bit priors, min-bit floors) for `qwen3_5_dense` / `qwen3_5_moe` / `deepseek_v4_flash`, plus the pure calibration prompt/metric helpers |

Provenance: a single `probe_evidence` artifact in, a single `optimizer_decision`
artifact out, carrying `source_probe_id`. The converter consumes the decision;
the runtime never reinterprets it.

## Building the artifact from probe evidence

`decide.decide(evidence, ...)` is the entry point. The flow:

1. **Adapt.** `_build_working_sets` reads each `probe_evidence` unit's *typed*
   fields (`kind`, `role`, `layer_index`, `projection`, `n_experts`, `shape`,
   `importance`, `quality`) and builds three working lists: `experts`,
   `affine_tensors`, `fp16_tensors`. No tensor-name parsing happens in this
   subsystem. The inventory already resolved every tensor's role, and the
   optimizer trusts those typed fields.
   - `shape` is the **true** `[out_features, in_features]` geometry. The
     sampled-row count lives elsewhere. Size formulas price bytes from `shape`, so conflating the two
     (recording sampled rows in `shape`) under-counts expert bytes ~2x and
     collapses any size budget.
2. **Build the q-tables through the monotone envelope** (see below).
3. **Optionally normalize expert importance** (`_normalize_expert_importance`,
   default `class-mean`): the probe's scalar importance is `mean(imatrix second
   moments)` of each tensor's *input* activations, and gate/up read the residual
   stream while down reads post-`SiLU(gate)*up` products ~23× smaller. The greedy
   compares these scalars across activation spaces, so raw values starve
   `down_proj` to 1-bit. Scaling each projection class to the global expert mean
   keeps the within-class layer ranking and makes the cross-class auction fair;
   the applied scales are recorded in `constraints`.
4. **Allocate** via `allocate.optimize(...)`.
5. **Render** the per-source allocation list (`_render_allocation`, sorted for a
   deterministic artifact) and **summarize** achieved metrics (`_summarize`:
   fidelity, worst-layer tail, sizes, bits-per-param, bit histograms).
6. **Gate** with `health_check`.
7. **Emit** via `make_artifact("optimizer_decision", ...)` with `objective`,
   `constraints`, `feasibility`, `allocation`, `achieved`, `rejected`, and
   `source_probe_id`.

### Infeasibility is a valid artifact

`allocate` raises `Infeasible` (the tail floor τ is unreachable even at max
bits) or `InfeasibleUnderBudget` (τ reachable but only above the size budget).
`decide` catches these and emits a **valid `optimizer_decision` describing the
infeasibility**: `status="invalid"`, `feasibility="infeasible"` /
`"infeasible_under_budget"`, an empty allocation, and a blocking validation
entry. The artifact-centered design wants the negative result durable and
inspectable too.

## The bit-allocation core (`allocate.optimize`)

Format- and name-neutral. It operates on working dicts carrying `layer_key`,
`importance`, `shape`, and a quality table; the adapter built those, so name
parsing never happens here. Allowed encodings:

```
EXPERT_BITS = [1, 2, 4]                    # TQ
AFFINE_BITS = [2, 3, 4, 5, 6, 8]         # affine MLX
GROUP_SIZES = [128, 64, 32]
```

Every affine tensor starts at its `min_bits` floor; every expert starts at
`EXPERT_BITS[0]`. Both phases only ever **lift** bits, never lower them, which
is exactly why starting at the floor preserves it.

### Phase 1, feasibility-first: worst-layer tail constraint (`_satisfy_tail`)

Only runs when a `tau` is given. It lifts bits until the worst-layer tail
`CVaR_α` over per-layer minima ≥ τ, so a single catastrophic layer can't hide
behind a good mean.

- **A layer's health is its weakest unit** (`aggregate.layer_minima`), and the
  tail (`aggregate.worst_layer_tail`) is the CVaR: the mean of the worst
  `ceil(α·N)` layers' minima.
- **Reachable-ceiling feasibility check first.** Before lifting, it computes the
  best achievable tail using each unit's best q over *all* tuples (for affine,
  any `(bits, gs)` through `affine_best_q`). If
  that ceiling is below τ it raises `Infeasible` immediately. Using the best of
  any tuple avoids wrongly declaring a τ infeasible that a different group_size
  would clear.
- **Each step**, it identifies the bottom-α tail layers, and for each tries to
  lift its argmin units (`q ≤ layer_min`). It bundles **tied** minima: lifting
  one argmin leaves the layer min unchanged, so all tied units must lift
  together. It recomputes the **non-separable** Δtail for each candidate bundle
  and picks the bundle with the best Δtail-per-byte.
  - An expert lift is the next bit up. An affine lift is the best
    quality-gain-per-byte *frontier* move, **including diagonal `(bits, gs)`
    moves**, so the tail can buy a cheaper or finer-gs tuple as well as the next
    bit at the current group size.
- If no tail layer has a productive upgrade, it raises `Infeasible` (stuck below
  τ).

Lifting bits to meet the tail *preserves* tail feasibility for the next phase,
because the greedy that follows only upgrades.

### Phase 2: importance-weighted fidelity-per-byte greedy

A best-first heap of upgrade moves, each ranked by **value per byte**:

- expert: `importance · q_gain / size_cost`
- affine: `objective_importance · q_gain / size_cost` (with optional
  destination-bit weighting; see role elasticity)

The greedy pops the highest-priority move, applies it if the unit is still at
the expected from-state (else re-pushes from the current state), accounts the
size/loss delta, and re-pushes the unit's next-best move. It stops the moment a
target is satisfied (quality reached or size filled). Because it only upgrades,
the Phase-1 tail feasibility is never broken.

**Affine frontier moves** (`affine_frontier_moves`) enumerate *all* measured
`(bits, gs)` tuples for a tensor (≤ 6×3 = 18), keep only those that buy quality
at positive size cost, then drop dominated targets (a candidate beaten by
another with `q ≥` it and `size ≤` it). Diagonal moves like `(4,128)→(6,32)` are
reachable in one hop. Only the single best-value-per-byte frontier move is
pushed per tensor (pushing all ~18 and re-pushing after every hop blew the heap
up to seconds per optimize on a 35B); the rest of the frontier is still explored
over time as the greedy re-pushes from each new state.

### The named objective (`decide._objective`)

The artifact records a human-readable statement of what the optimizer actually
maximized, never a value the greedy didn't optimize. The goal is always
`maximize importance-weighted fidelity F per byte` (or `maximize role-adjusted
affine risk reduction per byte` when a role profile is active), `s.t.` the active
constraints:

- `worst_layer_tail(CVaR_α) >= tau`
- `fidelity >= target_quality`
- `size_gb ~= target_size_gb (fill-toward objective)`
- a strict expert/affine `budget_split`

**`target_size_gb` is a fill-toward objective.** It does not impose a strict
`<=` budget.
The greedy upgrades bits per fidelity-per-byte until the package reaches (`>=`)
the target and stops, landing slightly *above* it. When multiple targets are
given, the **first satisfied stops** the greedy: e.g. with quality + size,
whichever is met first wins, so a small quality target can stop the run before
the size budget is spent. The objective string is worded honestly to match this
behavior (the earlier `<=` wording misdescribed it).

### Strict expert/affine budget split (`budget_split`)

A research lever, exposed on the convert CLI as `--expert-allocation-ratio`
(the fraction of the spendable budget reserved for experts). After paying the fixed
fp16/structural cost and the expert/affine bases, the remaining bytes are split
by fraction into an expert budget and an affine budget; experts spend only from
the expert side and affine only from the affine side, on two separate heaps. The
split is **strict** (leftover on a saturated side is recorded as unused, never
transferred), so the experiment can answer "what if this much budget is reserved
for experts vs affine?". Supports `target_size_gb` only; `target_quality` and
`tau` with split are rejected. The split (budgets, spent, unused) is recorded in
`achieved`.

### Codec decision parameters

`decide` also takes codec-level parameters, used by the DeepSeek-V4 path and
recorded in `constraints`:

- **Lossless mxfp4 carry** (default on). A routed expert whose source codec is
  FP4 (`fp4_e2m1_ue8m0`) can be carried losslessly as `mxfp4` at 4 bits, so
  the 4-bit rung is scored q=1.0 with codec `mxfp4` instead of TQ.
  `force_tq4_lossless=True` is the ablation switch that keeps 4-bit on TQ.
- **`force_dense_lossless_mx`**: a dense tensor whose `lossless_codecs`
  include `mxfp8` starts at `mxfp8` group 32 instead of the affine floor, so
  a lossless dense carry is the minimum rather than an upgrade target.
- **`min_routed_expert_bits`**: a floor on routed-expert bits (e.g. disabling
  the 1-bit rung for a serve-quality ablation); disabled rungs are recorded in
  `rejected`.
- **`expert_importance_norm`** (default `class-mean`): the cross-projection
  importance normalization described in the flow above.
- **`affine_role_profile_name`**: the role-elasticity profile identity
  recorded with the decision.

## Packed-size formulas (`sizes.py`)

Pure arithmetic (disk bytes per tensor at a given bit-width) used by the
optimizer to price each bit-up against the budget. Four formulas:

**Affine (MLX)** (`affine_bytes`):
```
weight_bytes = (rows*cols*bits + 7) // 8
n_groups     = rows*cols // group_size
overhead     = n_groups * 4          # fp16 scale + fp16 bias per group (2*2 bytes)
affine_bytes = weight_bytes + overhead
```

**TQ expert (TurboQuant)** (`tq_expert_bytes`):
```
packed_per_row = ((cols*bits + 31) // 32) * 4    # 32-bit packed words
per_expert     = rows*packed_per_row + rows*2    # + fp16 per-row norms
tq_expert_bytes = n_experts*per_expert + 1       # + 1 byte per-tensor tq_bits scalar
```

**Source-mxfp4 expert** (`mxfp4_expert_bytes`): per expert,
`rows * (ceil(cols/8)*4 packed bytes + one uint8 UE8M0 scale per 32 inputs)`;
no norms, seed, or affine biases.

**Dense MX float** (`mx_float_bytes`): `mxfp4`/`mxfp8` at fixed group 32, one
uint8 UE8M0 scale per group, no bias side table.

## Design rule: dense tensors are affine-only, TQ is expert-only

Non-expert (dense) 2D weights are quantized affine only; TurboQuant is used only
for routed experts. The split is hard because the two tensor classes have
different execution costs.

**Reason:** TQ carries a per-use decode penalty. For a routed expert that
penalty is paid only when that expert is actually selected for a token (sparse,
top-k). For a dense backbone weight, the tensor is used on **every token**, so
the TQ decode penalty would land on every forward step. Affine quant has no such
per-use decode cost, so dense weights are affine; the TQ decode cost is only
worth paying where activation is sparse: the routed experts. (The TQ allocation
itself is owned by the expert optimizer and is intentionally untouched by the
affine role-elasticity work.)

The narrower fp16 passthrough set follows structural requirements:
only `moe.router_gate` and `moe.shared_expert_gate` stay fp16, because they steer
discrete top-k routing: a decision a reconstruction-error proxy fundamentally
cannot score, so the optimizer must not assign them bits. SSM `in_proj_*` weights
use **affine** storage. They are projection weights; only routing gates stay fp16
(pinned by `test_decide.test_ssm_in_proj_a_b_are_quantized_not_fp16`; keeping
them fp16 made the runtime diverge from the reference and serve garbage).

## The allocation health gate (`health.py`)

A **pure, model-free** check: a list of rendered allocation dicts in, a list of
`Validation` out (empty == healthy). No model, no logits, no I/O.

It exists because the optimizer's objective (importance-weighted fidelity, even
with a CVaR tail) can be *satisfied* by an allocation that nonetheless serves
garbage. The serve cost of that is invisible to a reconstruction-error proxy.
The canonical trap is `--target-quality 0.95 --tau 0.9`: the tail passes
(`0.90 >= τ`) while hundreds of affine tensors sit at 3-bit, including lm_head, a
collapsed backbone. The gate inspects the *chosen bits* and rejects the
known-bad signatures **before** a multi-GB package is written. The four checks:

1. **Collapsed critical tensor**: a serve-critical single tensor below its
   floor (`lm_head` < 6-bit, `embed_tokens` < 4-bit). The CVaR tail cannot
   protect these because each is its own one-unit "layer".
2. **Collapsed backbone**: more than 10% of affine backbone tensors at ≤3-bit
   (the `--tau 0.9` trap). Expressed as a fraction so it scales with model size;
   skipped below 16 affine tensors so it can't false-positive on tiny synthetic
   allocations.
3. **Collapsed experts**: more than 50% of expert groups at ≤1-bit. Routed
   experts may legitimately use low bits, but an all-1-bit majority is
   unsupported. Same min-count guard (16).
4. **DeepSeek-V4 hash-routed expert floor**
   (`optimize.deepseek_v4_hash_expert_below_floor`): DS4 source-FP4 routed
   experts in the first three layers are hash-routed by token id, so learned
   top-k route averaging cannot protect them; a source-FP4 expert in a
   hash-routed layer allocated below the lossless 4-bit tier is blocking.

Thresholds are calibrated against known-coherent reference allocations, which
carry roughly one low-bit affine tensor, so they are not magic numbers. A
blocking finding makes the decision `status="invalid"` so a bad recipe
can't silently ship. `decide(allow_unhealthy=True)` is the explicit escape hatch:
the same findings are still recorded but downgraded to non-blocking warnings (plus
an `optimize.health_overridden` info entry), and the decision stays valid.

## lm_head / embed safety floors

`decide` applies serve-critical min-bit floors **regardless of the tail
threshold**: `lm_head` (default 6-bit, caller-tunable via `lm_head_bits`) and
`embed_tokens` (4-bit). These are SINGLE global tensors (each its own one-unit
"layer"), so the CVaR worst-layer tail can never protect them; a quality/τ recipe
would otherwise strand them at the bit floor.

They are a structural-safety constraint rather than part of the optimization
objective, and they hold independent of τ. Because both allocation phases only
ever lift bits, seeding `cur_bits` from `min_bits` preserves the floor while the
tail constraint and greedy upgrade further on top.

The split of responsibility is deliberate: the **backbone** is protected by the
health gate, *not* by per-role floors. Broad floors recreate a "uniform-high"
mistake that destroys mixed affine. Only the two single-tensor landmines get hard
floors.

## Monotone-in-bits quality envelope (`monotone.py`)

The probe measures `q` on sampled rows, so a higher-bit encoding can occasionally
score slightly **below** a lower-bit one: pure sampling noise. A
fidelity/risk-per-byte greedy would happily exploit that to justify *fewer* bits.

Before optimization, `decide` projects each unit's q-table onto a monotone
envelope (`monotone_envelope_by_bits`): walking bits ascending, replace each `q`
with the running max so **more bits never yields less q**. Values are only ever
raised; an already-monotone table is unchanged. Three q-table key shapes pass
through it: expert `{bits}`, affine `{(bits, gs)}`, and dense MX
`{(format, bits, gs)}`. For the affine and dense-MX shapes this is enforced in
`bits` **independently per group_size** (the bits ladder is the precision
axis; gs is a separate knob). This makes the noise exploitation
impossible and keeps "spend a bit, get less risk" true, which the greedy relies
on.

Genuine inversions are **reported, never hidden**. `non_monotonic_inversions`
flags adjacent pairs whose higher-bit q drops more than a 0.05 noise band; for
each, `decide` emits a **non-blocking** `optimize.q_inversion` warning ("likely a
sampling artifact; the envelope smoothed it for allocation, but consider
re-measuring"). The envelope already made the allocation safe, so the warning
never blocks. It surfaces a measurement smell for investigation.

## Affine role elasticity (`affine_elasticity.py`)

The default affine scoring is tensor-local (`importance · q_gain / size_cost`),
which can starve residual-write roles that look locally fine but have high
downstream leverage. Role elasticity layers a calibrated, evidence-backed prior
on top, without changing the streamed conversion, package, or runtime path: only
the scoring weight changes. The three knobs, all wired through `decide` and
consumed by `allocate`:

- **`affine_role_weights`** → `objective_importance` (general upgrade pressure
  for a role; protects `ffn.down_proj` / `ssm.out_proj` / `attn.o_proj`,
  discounts elastic SSM input roles).
- **`affine_role_bit_weights`** → `objective_bit_weights` (priority-only bias on
  *which destination bit-width* is worth buying; reported fidelity still comes
  from the original q-table).
- **`affine_role_min_bits`** → per-role start floors (bit priors are upgrade
  preferences. A separate min-bit floor forbids unsafe
  *starting* bits, proven necessary for transfer to the 4B model).

These are calibrated from small full-model runs (Qwen3.5-0.8B / 4B behavior) and
transfer an expected **direction**. The target model's own streamed q-table
remains authoritative and can overrule the prior. They are applied as a family
convention (no CLI flag): `qwen3_5_dense` uses `qwen35_affine_role_band_v1`; `qwen3_5_moe` uses a
distinct `qwen35_moe_affine_role_band_v1` that reuses dense attention/SSM priors,
aliases MoE shared-expert affine roles to the dense FFN priors, keeps routers
fp16, and leaves routed-expert TQ untouched. `deepseek_v4_flash` uses a
dense-conservative profile: DS4's routed experts carry the aggressive
compression, so non-expert affine tensors are backbone infrastructure and hold
a 6-bit floor with destination-bit priors centered on 6-bit. The profile
identity is recorded in the decision's constraints/provenance, and a
missing/absent profile falls back to current behavior (fail-safe). The module also holds pure calibration helpers
(prompt set, scoring, Q8-relative summaries) so the behavior harness stays
testable without loading MLX.

## Aggregate proxy scalars (`aggregate.py`)

Pure, format-neutral, model-free, they score an allocation cheaply from the
probe's precomputed q-tables, never from weights or a model:

- **`fidelity(units)`**: importance-weighted mean quality over `(importance, q)`
  pairs; the calibrated optimization target F. Falls back to the unweighted mean
  when total importance is non-positive.
- **`layer_minima(units)`**: worst (min) q per layer over `(layer, q)` pairs.
- **`cvar(values, alpha)`**: mean of the lowest `ceil(α·N)` values (expected
  shortfall); `α=1` is the plain mean, `α→0` approaches the minimum.
- **`worst_layer_tail(units, alpha)`**: the hard-constraint statistic T_α: CVaR
  over per-layer minima. Averaging the worst few *layers* means a single
  catastrophic layer cannot hide behind a good overall mean.

## Where the code lives

The numerical core and its tests are the authority:

- `src/moespresso/optimize/allocate.py`: the two-phase allocation core.
- `src/moespresso/optimize/decide.py`: the adapter, objective string, floors,
  health-gate wiring, and infeasibility-as-artifact.
- `src/moespresso/optimize/health.py`, `monotone.py`, `sizes.py`,
  `affine_elasticity.py`, `aggregate.py`: the supporting pure modules above.
- `tests/test_decide.py`, `test_allocate_frontier.py`, `test_health.py`,
  `test_monotone.py`, `test_aggregate.py`: the behavior these modules guarantee.
