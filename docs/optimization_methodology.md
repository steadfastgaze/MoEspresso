# Optimizing a served model: design, tooling, and methodology

Scope note: this document distills one serving-optimization effort
(DS4-Flash). Its measurement discipline transfers, while the levers,
instruments, bars, and their ordering do not. A new model family starts from
its own goals and constraints
and re-derives all of those by measurement; the Ornith work is the
recorded example of the DS4 recipe failing to transfer while the
method held.

The DS4-Flash serving path moved from 15.7 to 26.2 tok/s decode while
its quality gates improved. This document is the transferable method
rather than the individual tricks. The intended reader is an engineer
or agent starting the same climb on a different model architecture.
It sits at the level above the model-specific findings: what to build
first, how to measure, how to decide, and where speed generally comes
from. Treat every architecture detail below as an example of a
category.

## 1. The frame

Serving optimization is closing a gap to a reference number under a
hard quality constraint. Fix the three coordinates before touching any
code:

- A reference: a faster engine on the same hardware and weights, or a
  hardware roofline estimate. Without one there is no gap, only vague
  ambition. Run the reference yourself, in the same session as your
  own runs, on the same prompt.
- Two speed numbers: prefill (time to first token on a long prompt)
  and decode (tokens per second at steady state). They have different
  physics (throughput versus latency, GEMM-bound versus
  dispatch-bound) and different levers. Track them separately.
- Quality anchors: numeric bars that must never regress. Define them
  before the first speed change, because every later decision is
  judged against them.

## 2. Quality is token and logit identity

Text can read fine while logits are wrong, so plausible output proves
nothing. Every quality instrument
must be numeric:

- A tiered ladder, cheapest first: exact token identity against a
  recorded run; teacher-forced negative log likelihood over long
  prompts (thousands of scored tokens); task-level gates with exact
  expected outputs. Cheap tiers run on every change; expensive tiers
  run for math-affecting changes.
- Record token rails: the exact token ids a fixed prompt produces
  under a named configuration, committed to the repo. Every candidate
  change replays the prompt and compares ids. Two identical runs
  (an A/A pair) prove the rail is deterministic before any A/B is
  trusted.
- Expect the knife edge. A low-bit quantized model sits near decision
  boundaries, and any valid change to floating-point evaluation order
  flips some tokens deterministically. A flipped token is not by
  itself a regression. Distinguish a realization flip from a real loss
  with the NLL tier: if teacher-forced NLL over thousands of tokens is
  flat or better while a downstream count moved, the count moved on a
  knife edge. Judge kernel-numerics changes by NLL, never by which
  tokens happened to win.
- One caution the ladder taught: a cache-less scorer can miss the
  decode path entirely. If the fast path only engages with a KV cache
  and single-row inputs, a full-forward scorer certifies the wrong
  code. Build an engaged probe: prefill the prompt head as one cached
  chunk, then teacher-force the tail one token at a time through the
  served decode route, with counters proving the route ran.

## 3. Measurement discipline

Measurement error caused most of the wasted effort in this work. The rules that
survived:

- Measure the served shape. Decode microbenchmarks are only evidence
  when seeded to the served steady state: a real prefill, a dozen
  settle steps verified against the token rail, then measurement with
  real captured module inputs (install a temporary hook, capture the
  live tensors, restore the hook). Random inputs at the wrong shapes
  measure a different program.
- Fenced stage timing: wrap the candidate stage in a build-plus-sync
  timer, alternate implementations in blocks (never per call), and
  calibrate the fence cost in the same loop. A stage that prices at
  the fence reading is floor-priced; stop optimizing it.
- The concurrency trap: a fenced win can vanish in serving because the
  pipeline overlaps the stage you shrank with other work. Fenced wins
  justify building; only served wins justify landing. The landing bar
  is four alternating served arms, one process each, where every ON
  arm beats every OFF arm with no overlap.
- Arms must be proven different. Export engagement counters from every
  route and read them per arm. A null result with unproven arms is a
  recorded trap: both arms silently ran the same code.
- The metric must be independent of the change. In-process whole-token
  deltas measured inside one resident session did not convert to
  served cold-process wins here; treat them as pre-signals only.
- Environment gating: absolute numbers require a thermal state check
  (a headless sensor read before every arm), the same power source
  (battery versus AC invalidates cross-arm comparison), and an
  otherwise idle machine. Interleave your arms with the reference
  engine's arms in one session and compare matched pairs; ambient
  drift moves both engines together, so the matched pairs carry the
  comparison and the day-to-day spread carries nothing.
- Measure or revert. Every change declares its numeric bar before the
  measurement. Below the bar, it reverts with a short note. The note
  matters: parked mechanisms get re-priced when the stack changes, and
  several of this project's largest wins were previously parked ideas
  whose economics a later change reversed.

## 4. Attribution before levers

Do not collect tricks; build a ledger. An attribution of record is a
per-stage table of the token (or the prefill pass) that reconciles to
the end-to-end number, with any residual named explicitly. The ledger
does three jobs:

- It ranks levers by recorded size, so effort goes to the biggest
  line instead of the most interesting one.
- It exposes lies: if the stage sum does not reconcile to the served
  rate, the measurement method is broken somewhere.
- It goes stale on purpose: after each landed change, the ledger must
  be refreshed, because a route change re-ranks everything.

Keep two standing documents beside it: a dead-end stop-list (every
parked idea with the number that killed it and the mechanism), and a
periodically re-consolidated operating manual (current verdict, lever
queue, acceptance rules, decision tree). Long optimization campaigns
die of forgotten evidence; the logs are the memory.

## 5. Where speed comes from

The individual kernels differ per architecture; the categories do not.
In observed order of value on this project:

- Dispatch granularity. Decode is dispatch-bound: at interactive
  rates the per-token budget is tens of milliseconds and every kernel
  launch and Python transition costs real fractions of it. Count
  dispatches per token first. Fuse ops that always run together
  (projection plus activation, split plus weighted sum), batch
  weights instead of looping (a grouped projection wants one batched
  or gathered dispatch where a naive port writes N slices plus a
  concatenate), and keep one dispatch boundary per composite op. The reference engine's shape is
  a target: it ran roughly twenty fat dispatches per prefill layer
  where this engine ran many more.
- Serve from the format you store. Quantized weights can be consumed
  three ways: dequantize then matmul (a bridge; pays materialization
  every call), repack losslessly into a form a native fast kernel
  accepts (free when an exact mapping exists), or run a kernel
  directly on the wire bytes. Which wins depends on shape: single-row
  decode matvecs and bulk prefill GEMMs prefer different forms.
  Measure per shape class rather than per tensor. Decode-quality logic also
  differs by role: a dense weight pays its decode penalty on every
  token, a routed expert only when selected, so dense tensors want
  cheap exact decode paths and experts tolerate heavier codecs.
- Precision seams. Know where full precision is protective (norms,
  logits, residual-stream seams) and where a reduced-precision
  activation lattice is acceptable. Half-precision output seams were a
  recorded dead end here; bfloat16 activation lattices inside decode
  matvecs passed the full ladder twice and even improved long-prompt
  NLL. There is no rule that lower precision costs quality; there is a
  rule that any precision change is math-affecting and must run the
  full campaign.
- Memory and residency. Duplicate representations (wire bytes plus a
  repacked form) are a legitimate measured tradeoff; keep each copy
  only while it has a consumer. Build anything off the default path
  lazily: an eagerly constructed fallback representation cost this
  project gigabytes of RAM and, less obviously, seconds of first-token
  latency, because construction ran inside the first prefill. Prewarm
  whatever selects the fast path (expert residency here) so that runs
  are deterministic and the fast path is pinned from the first
  request.
- Accumulation order. Bit-identity across a kernel swap requires
  replicating the original kernel's reduction order exactly, which is
  possible (transcribing the framework's internal reduce loops) and
  valuable: order-preserving changes certify with token identity
  alone, order-changing changes buy the full campaign. Decide which
  kind you are writing before you write it.

## 6. The change contract

Every landed change ships with the same five parts, and the discipline
is what keeps fifteen stacked levers debuggable:

- A kill switch: an environment variable that restores the previous
  route exactly, plus a family switch that closes a whole class of
  routes at once. Remove a per-route switch only after certification,
  as a deliberate simplification.
- Fail-closed eligibility: the new route checks every contract it
  depends on (dtype, dimensions, block sizes, kernel availability) per
  call and falls back to a correct path on any mismatch. Fallbacks are
  how an off-contract package serves correctly instead of crashing or,
  worse, silently corrupting.
- Engagement counters: per-route call counts exported through one
  stats surface, so any run can prove which code executed.
- Tests: engagement, fail-closed behavior on each contract violation,
  switch behavior, and a numeric bound against the exact reference
  form.
- A log entry with the numbers: mechanism, fenced and served results,
  ladder results, rail status, and the verdict.

## 7. The campaign protocol

For a math-affecting change, in order, stopping at the first failure:

1. Fenced A/B at the seeded steady state: is the win real in
   isolation.
2. Bounded served checkpoint (a few tokens): first ids on the correct
   rail, counters at expected values. Cheap, catches wiring mistakes
   before long runs.
3. Cache-less quality gate (fast, broad).
4. Engaged teacher-forced NLL through the touched path, ON versus OFF
   arms, counters proving the swap, a pre-declared regression bar.
   Run each arm in its own process; thousands of single-token
   evaluations can exhaust per-process GPU event pools.
5. Task-level gates.
6. Four served alternating arms; every ON beats every OFF.
7. Rail decision: token-identical means no fork; a fork (expected for
   math-affecting changes) means storing a new A/A-verified rail for
   the new default and keeping the old rail for the kill-switch
   configuration.
8. Log entry, then commit.

Bit-identical changes skip to token identity plus the bounded
checkpoint. The protocol looks heavy; in practice it is roughly
an hour wall-clock, most of it unattended, and it caught real problems
at every tier at least once.

## 8. Tooling to build first

The campaign is only as fast as its instruments. Build these before
optimizing anything:

- A speed-stats CLI: load the package, run a rendered prompt with a
  token cap, emit JSON with time to first token, decode rate, the
  generated token ids, and every engagement counter. This one tool is
  the bounded checkpoint, the served arm, and the rail check.
- A quality-gate CLI with the ladder tiers as subcommands, pinning
  whatever state (residency, caches) makes runs deterministic.
- A committed reference-runs directory holding the rails and the
  certification copies, named by configuration.
- A probe-script pattern: seed, assert the settle trace on-rail,
  capture real inputs via temporary hooks, fence in alternated blocks,
  write JSON. Every new question copies the pattern.
- A one-line thermal and power gate command, run before every
  absolute number.
- Status files for anything long-running: append resume state after
  every step so an interrupted campaign continues instead of
  restarting. Trust results only from artifacts on disk after
  confirmed process exit.

## 9. Using the reference engine

The reference is an oracle and a pace-setter as well as a target
number. Two uses beyond the scoreboard:

- Byte-faithful import: copying the reference's quantized bytes into
  your own package isolates encode quality from kernel quality. This
  project found its own re-encode was functionally worse (measurably,
  on NLL) than the reference's bytes at equal weight-space error;
  judging encoders by reconstruction error alone is a recorded
  mistake. Judge encoders functionally.
- Same-session interleaved certification: alternate your arms with
  reference arms on the same prompt within minutes of each other, and
  publish matched pairs. Claims of parity or superiority made any
  other way do not survive ambient drift.

## 10. Process

- One lever at a time, committed only after its campaign, so any
  regression bisects to one change.
- Park loudly, with the killing number and mechanism written down, and
  re-price parked items whenever a landed change moves their
  economics. The decode route that finally beat the reference here was
  assembled from two previously parked observations.
- The ledger, the stop-list, and the log are not documentation
  overhead; they are the optimization state. An agent (of any size)
  resuming from them should be able to pick the next lever without
  re-deriving anything.
- Quality-first is a product identity. Every speed claim in this
  project is conditional on anchors that never moved down; that is
  what makes the claims usable.
