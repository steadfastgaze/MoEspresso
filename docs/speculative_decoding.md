# DeepSeek-V4-Flash speculative decoding

Speculative decoding runs a small draft model that proposes a short block of
tokens, verifies the block in one multi-token target forward, and keeps the
longest accepted prefix plus a corrected or bonus token from the target.
Single-token decode is weight-traffic bound, so verifying several tokens in one
forward amortizes the routed-expert reads. The loop is lossless by rule: with
temperature 0 a draft token is accepted only when it equals the target argmax
at its position, and with temperature above 0 the standard speculative sampling
rule (accept with probability `min(1, p_target/p_draft)`, resample the first
rejection from the residual distribution) recovers the target distribution
exactly.

Correctness semantics, the verify-lattice property, and the acceptance
evidence are in [`deepseek_v4_quality.md`](deepseek_v4_quality.md). Verify-path
costs, kill switches, and the measured results are in
[`deepseek_v4_speed.md`](deepseek_v4_speed.md). Sidecar builds are in
[`deepseek_v4_package_recipe.md`](deepseek_v4_package_recipe.md); the sidecar
manifest kinds are in [`package_format.md`](package_format.md).

## The drafter protocol

`runtime/deepseek_v4/spec_decode.py` defines the loop and the `Drafter`
protocol every draft model implements:

- `block_size`: the maximum tokens proposed per round;
- `tap_layer_ids` and `tap_transform`: which target layers to tap and how each
  tapped hidden state is shaped at record time;
- `make_state` and `ingest`: opaque per-generation draft state, fed the tapped
  rows and committed token ids in stream order;
- `draft`: propose up to `block_size` tokens after an anchor, returning fp32
  draft logits and, when the drafter has a confidence head, raw per-position
  confidence logits;
- `greedy_only` (optional, read with a default of False): set by a drafter
  whose proposals are argmax-only with no draft distribution over the target
  vocabulary. The loop refuses temperature > 0 for such a drafter and the
  serve seam keeps those requests on the plain path.

`spec_generate` drives any drafter. Chunked prefill feeds tap rows to the
drafter as they are produced. Each round drafts a block, runs one target
forward over the anchor plus the proposal, applies the acceptance rule, and
restores the target caches to the accepted frontier from a pre-verify snapshot
(`dspark_rollback`). The verify pass reads hidden states from the inner graph
and applies the fp32 language-model head itself, because the served head patch
slices multi-token forwards to the newest row and verification needs every row.
The proposed tokens stay on device and the acceptance rule evaluates the
argmax and tokens together, leaving one host sync per round. Once a
round's acceptance is decided and the accepted rows are ingested, the
loop builds the next round's draft graph and dispatches it
asynchronously: the draft reads only the drafter state, so its kernels
execute while the host performs the round's bookkeeping, streaming
callbacks, and the next round's cache snapshot. The proposals are
identical to the sequential schedule.

## Drafter families

The runtime contains DSpark and DFlash family implementations. The public
external-sidecar command accepts DSpark. DFlash and the quarantined MTP
implementation remain available to future model packages and internal
engineering tools.

**DSpark** (`runtime/deepseek_v4/dspark_model.py`) is the semi-autoregressive
drafter shipped with the DeepSeek-V4-Flash-DSpark checkpoint: three MoE draft
blocks attending over a 128-token sliding window, conditioned on the mean over
hyper-connection copies of the target's layer 40-42 outputs, concatenated. A
sequential Markov head biases each position's logits from the previously
sampled draft token, and a confidence head emits one raw logit per position;
consumers apply the sigmoid. Block size 5. Its per-round confidence
discrimination is its edge on open-ended content.

**MTP** (`runtime/deepseek_v4/mtp_model.py`) is retained research code for
future checkpoints, not a released drafter path. It has no installed builder,
serving selector, replay selector, or battery option. No valid sidecar for the
current weights exists: the
implementation targets preview-era `mtp.0` semantics (an `e_proj`/`h_proj`
head) that the 0731 checkpoint does not carry, and its own sidecar builder
refuses that checkpoint. It also carries two known draft-tree graph defects
that are documented rather than fixed: the mHC recombine inherits the stock
vendored orientation where the reference contracts over the first
hyper-connection axis, and the compressed-KV non-RoPE rows are stored
unrounded where the reference rounds through E4M3FN. The same pair was fixed
in the DSpark draft tree; the trunk serving graph was never affected. Every
acceptance or decode number measured through this drafter is invalid until
those defects are fixed, so none is published here or in
[`deepseek_v4_speed.md`](deepseek_v4_speed.md). Greedy token identity was
never at risk on any of this: speculative decoding accepts by verification
against the target. The graph chains the base DeepSeek-V4-Flash checkpoint's
multi-token-prediction layer with an additive fusion,
`e_proj(enorm(embed(token))) + h_proj(hnorm(h))`, where `h` is the raw
hyper-connection hidden state after layer 42; one vendored decoder block
(sliding window 128) and the shared language-model head follow. The single
vendor-trained module chains to a depth cap of 3. There is no confidence
head, so scheduling uses observed acceptance.

**DFlash** (`runtime/deepseek_v4/dflash_model.py`) is the block-diffusion
drafter from the RedHat DeepSeek-V4-Flash speculator checkpoint: five dense
llama-type draft layers conditioned on the raw hyper-connection streams of
five target layers. The checkpoint's aux ids `[3, 13, 23, 32, 42]` index the
HF hidden_states convention where entry 0 is the embedding output, so the
tapped 0-based decoder layers are `[2, 12, 22, 31, 41]`. Each committed
token contributes one fused feature row, `hidden_norm(fc(concat))` over the
stream-major flattened taps, projected into per-layer context K/V at its
absolute position; the context cache keeps the trailing 2048 entries
(the sliding window, measured back from the anchor), which bounds the
per-request draft cache near 10 MiB at the shipped dimensions. A round embeds
`[anchor, mask x 7]` through the target embedding and runs a single
non-autoregressive pass, causal within the block, over the windowed
context; the in-block K/V stay in scratch and rejected rows never enter
the cache. Draft logits cover a pruned 32000-entry vocabulary and the
drafter carries its own pruned head; proposals map to target ids through
the d2t offset table. Proposals are argmax-only with no draft distribution
over the target vocabulary, so DFlash engagement is greedy-only: a
temperature > 0 request takes the plain path. Block: 7 proposed tokens
per round. With temperature 0 the loop's guarantee is unchanged: no
unverified token is ever emitted, every kept token equals the target's
own argmax on the verify forward. As with the other families, the
multi-token verify and single-token decode are different numeric
lattices, so knife-edge argmax flips against a plain stream are possible
and reported by the replay harness.

DSpark shares the target's embedding and language-model head. DFlash shares
the embedding and carries its own pruned head. No released sidecar stores an
embedding.

## Memory

| Component | Size |
|---|---:|
| DSpark sidecar, resident | 10.2 GiB |
| Lazy q8_0 affine views | ~3.1 GB |

The q8_0 affine views back the batched fp32-seam route for verify-shaped
forwards and materialize lazily on the first multi-row verify.

The K-quant target with the DSpark sidecar peaks near 96 GiB, which sits at the
default wired ceiling on a 128 GB host; raise `iogpu.wired_limit_mb` before a
full-resident run. The shipping package's certified request peaks, with and
without its bundled DSpark drafter, are in
[`deepseek_v4_speed.md`](deepseek_v4_speed.md).

## The adaptive scheduler

Each round chooses how many draft tokens to submit for verification. The
scheduler bakes in no constants; everything is calibrated online per
generation:

- **Round cost model.** An exponential moving average of measured wall time
  per submitted length, seeded by a prior anchored on the measured plain
  decode step. Once two lengths are observed, the fixed and marginal costs are
  fitted from the observations, because static cost priors go stale after
  kernel changes.
- **Confidence calibration.** A confidence head's raw sigmoid levels are
  unreliable as absolute probabilities (measured pessimistic by roughly two to
  one on high-acceptance content) while its per-round ranking is useful. The
  calibrator rescales each round's predictions by the observed-to-predicted
  acceptance ratio per position. A drafter without a confidence head reports a
  neutral prediction, so the calibrated survival reduces to the observed
  acceptance ratios.
- **Exploration.** The first rounds submit the full block, and periodic probe
  rounds alternate full and half blocks, because a scheduler that always
  shortens rounds starves its own acceptance and cost statistics.
- **Plain-decode floor.** The chooser maximizes expected emitted tokens per
  millisecond over submit lengths including length 0, a plain decode step, so
  a round where speculation is not expected to pay falls back to plain
  decoding.

A fixed confidence-threshold truncation also exists (`confidence_threshold`,
default 0, off). Its constants are deliberately unset: thresholds tuned
against the current confidence head would be weight-dependent.

## Running it

The battery runs a fixed ten-prompt evaluation set through the plain arm
and each configured drafter in one process, with repeat aggregates and
thermal provenance in the report; it is the standard measurement
instrument. The prompt set is evaluation-only and never feeds tuning.

```bash
uv run --locked moespresso-ds4-spec-battery \
  --package <package-dir> \
  --dspark-sidecar <sidecar-dir> \
  --dflash-sidecar <sidecar-dir> \
  --json-out <report.json>
```

The replay harness is the single-prompt entry point:

```bash
uv run --locked moespresso-ds4-dspark-replay \
  --package <package-dir> \
  --sidecar <sidecar-dir> \
  --drafter dspark \
  --prompt-file <prompt.txt> \
  --json-out <report.json>
```

`--drafter` selects `dspark` or `dflash`. The harness runs the same
prompt through plain greedy decoding and speculative decoding with a shared
prefill schedule and reports token identity with the first divergence index,
acceptance statistics, per-position and submit-length histograms, and
wall-clock speed. `--no-adaptive` fixes the verify length at the drafter block
size, `--temperature` selects the sampled acceptance rule (rejected for the
greedy-only DFlash drafter), and `--confidence-threshold` applies the fixed
truncation.

`moespresso serve <package> --drafter <sidecar-dir>` selects an external
DSpark sidecar. `moespresso generate` accepts the same option, and
`moespresso verify <package> --drafter <sidecar-dir>` checks the package and
sidecar together before serving. The command detects the family from the
manifest at the supplied root, accepts DSpark, and refuses DFlash or MTP in
this release. The sidecar loads once at model-load time. A sidecar that fails
to load refuses startup. The explicit command option overrides both the
environment variable and bundled automatic selection.

`MOESPRESSO_DS4_DRAFTER=off` remains the kill switch. The lower-level runtime
selector retains the existing family-specific values for internal tooling.
An explicit value in either direction is recorded as an override on the
drafter policy attestation.

An absent or empty variable selects automatically. A package whose
manifest declares a bundled `drafter` component (family `dspark`) takes
the capacity-gated path; the drafter enables only when all of the
following hold:

- the package is a DeepSeek-V4 family target;
- the runtime holds every routed expert of every layer resident: a pooled build
  whose live pools each hold the full expert set. Any bounded or streaming
  capacity below the full expert count resolves to off;
- every declared component file is present in the package directory. A
  distribution shipped without the optional component serves plain with
  the absence counted; a partially present component serves plain and
  names the missing files;
- the machine passes the wired-budget capacity check
  (`runtime/deepseek_v4/drafter_policy.py`): resident weights plus drafter
  files plus KV-and-pool state at the served 128k context plus the
  per-request working-set margin must fit the usable wired budget (the
  `iogpu.wired_limit_mb` sysctl when set, else the Metal device's default
  recommended working set). The margin is capacity-aware over the
  sorted-route split: the committed parts-16 working set (14.1702 GiB) is
  preferred, and when the drafter fits only at the parts-32 floor
  (11.6448 GiB, priced at -5.6 percent prefill with decode unchanged) the
  policy raises the process split default to 32 before declaring off,
  recording the selection (`sort_nsplit`, `sort_nsplit_source`). An
  explicit `MOESPRESSO_DSV4_IQK_SORT_NSPLIT` pins the split: the policy
  reserves that setting's own measured working set and never overrides
  the operator. Any unreadable term resolves to off;
- the bundled sidecar loads and validates.

The decision payload (mode, decision, reason, and every byte count that
drove the comparison) attaches to the served model and exports through
`iqk_engagement`, `ssd_streaming_stats`, and the speed-stats count keys
(`ds4_drafter_policy_auto_on`, `ds4_drafter_policy_auto_off`,
`ds4_drafter_policy_override`).

A package without a declared drafter component serves plain unless the user
passes `--drafter`. The declaration is the whole of automatic selection: no
directory is searched.

Any automatic miss prints one line naming the reason (`spec: auto off
(bounded residency)`, `spec: auto off (optional drafter component absent,
...)`, `spec: auto off (drafter budget: budget-exceeded)`, `spec: auto off
(package declares no drafter component)`, or `spec: auto off (sidecar failed:
...)`) and serving stays plain; automatic selection never refuses startup. The
drafter load prints the sidecar manifest's artifact id for provenance. A
sidecar paired with mismatched target weights costs acceptance only, never
correctness: verification compares every draft token against the target's
own logits, so a wrong pairing lowers the accepted length and the output
tokens do not change.

The served speculative schedule is family-keyed: the DSpark chain serves
the fixed:3 submit schedule (the schedule sweep's measured pick and the
drafter-on certification's configuration); other families serve the
adaptive schedule. `MOESPRESSO_DS4_SPEC_SCHEDULE` (`fixed:<K>` or
`adaptive`) overrides, failing closed on any other value, and the served
schedule is recorded in every response's `speculative` block.

The runtime truth line records the resolved state as one token:
`spec=dspark(auto)` for automatic selection, `spec=dspark`, or `spec=dflash`
for an explicit selection, `spec=off` for an explicit off, or
an automatic off reason (`spec=off(auto:bounded-residency)`,
`spec=off(auto:drafter-absent)`, `spec=off(auto:budget)`,
`spec=off(auto:no-declared-drafter)`, `spec=off(auto:sidecar-failed)`). A
residency signal that cannot be read fails closed to
`spec=off(auto:residency-unknown)`.

A request engages speculation only when the effective sampler is greedy
(temperature 0) or pure-temperature sampling. Top-p, top-k, and min-p
shaping, presence penalties, and per-token logprob requests bypass to plain
decoding unchanged, because the acceptance rule requires the draft
distribution used in `p_target/p_draft` to be exactly the proposing
distribution, shaping included. A drafter that advertises `greedy_only`
(DFlash) restricts engagement further to temperature 0: it has no draft
distribution to feed the sampled rule, so its temperature > 0 requests
bypass to plain decoding as well. Responses that took the speculative path
report the drafter family, rounds, acceptance, plain-decode fallbacks, and the
submit-length histogram under `usage.moespresso.speculative`.

## Cache reuse

The shipped DSpark runtime has a portable state capsule, so its eligible
requests use both in-memory prefix reuse and the disk KV tier. DFlash and any
DSpark implementation without the complete state protocol still serve from
a fresh per-request cache. For those non-resumable speculative paths, the
prompt-cache store and disk KV tier are bypassed and one notice is logged.

The in-memory store separates plain and speculative producers. The plain rail
contains ordinary target caches. A speculative rail identifies the cache
schema, numeric producer lattice, drafter family, sidecar artifact, and resolved
schedule. A DSpark entry pairs the target cache with a capsule containing the
drafter's projected-history rings and scalar state at the same public token
frontier. The target and companion bytes count together under the configured
in-memory cache budget. Cache lookup probes both rails without moving either
entry and selects the deepest reusable target. A compatible companion resumes
speculation; a target without one still provides plain suffix generation.

The draft/verify loop publishes only committed public state. Rejected proposal
rows and transient verify frontiers never enter either cache tier. Before
resuming, the runtime checks the producer rail, target frontier, companion
frontier, sidecar artifact, capsule kind, and capsule schema, then imports the
capsule through the live DSpark model. A failed check cannot fail the request.
The valid target is retained for plain generation when its frontier still
matches.

Disk checkpoints use the same split contract. The generic target checkpoint is
written and restored first. An optional DSpark companion is stored separately
and is accepted only for the exact target identity and producer rail. During
speculative prefill, a paired callback runs only at configured aligned
frontiers after both target caches and DSpark state report that frontier. It
commits the target before the companion. An attachment failure therefore never
invalidates a target checkpoint. See `disk_kv.md` for persistence, eviction,
quarantine, and visibility details.

An exact whole-prompt cache entry cannot itself resume generation because it has
an empty suffix and the cache does not persist the next-token logits. The
runtime does not load that exact entry's DSpark companion. The DSpark dual-rail
probe does not move an exact in-memory entry. A different, shorter compatible
target may still serve its non-empty suffix. If none exists, the request
performs a full prompt prefill and reports `exact_fallback`.

## Speculative decoding and SSD streaming

The speculative results assume a fully resident target. A verify forward
reads the union of experts routed across the block, which costs about one
token's routed weight traffic only when those experts are resident.

Draft predictions do not help the SSD-streaming tier prefetch experts. Deep-
layer routing depends on accumulated contextual state, so its predictable
component is continuity across nearby tokens, which the recency-based demand
cache already captures. The drafter's tap rows describe the committed context,
and draft-model activations align with output tokens rather than with the
target's per-layer router inputs, so they carry no routing signal for experts
the demand cache does not already hold.
