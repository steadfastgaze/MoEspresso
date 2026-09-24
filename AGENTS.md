# Agent and contributor guide

Read this before changing anything. `DEVGUIDE.md` and `docs/` cover the
architecture.

## Public product and fixture boundary

DeepSeek-V4-Flash and Ornith are the public product models. Ornith uses the
`qwen` architecture adapter. Qwen3.8-Flash-Next uses the separate `qwen4`
adapter with composite state, a KVarN attention cache and PLE storage.
Architecture support and engineering commands do not declare a published
model package.

DeepSeek-V4 fixtures are separated by provenance, regardless of ladder rung.
Keep published antirez/ds4 material committed so public gates can use it.
Provider-captured Q1 and Q2 continuations, selected-token records and
top-logprob payloads belong under
`src/moespresso/correctness/fixtures/deepseek_v4/private/`. Keep Q4 teacher
and candidate logit dumps and their scored marker set there too. Reference the
third-party WikiText corpus by digest without copying it. Consult
`docs/deepseek_v4_quality.md` for each fixture's boundary.

Keep unpublished Ornith benchmark questions, answer keys and verification
programs under `src/moespresso/correctness/fixtures/ornith/private/`. Neither
private fixture set belongs in source, tests, logs, documentation, wheels or
source distributions.

The public Ornith gate covers project-owned agentic-coding and long-context
instruments:

```
uv run --locked moespresso-ornith-gate <package> \
  --families agentic_coding,long_context
```

The full hard-reasoning gate also requires ignored private Ornith fixtures.
Source-release tests inject synthetic questions and keys.

## Private research location

Put investigation plans, campaign logs, measurements, throwaway probes and
research worktrees in sibling `moespresso-dev-support-private`. Do not recreate
`.research/` or `specs_archive/` here. Product implementation, tests and public
documentation stay in this repository.

## Tooling discipline

Use `uv` for everything. `uv sync` installs the runtime and development
environment, and tests and lint run lock-strict with `uv run --locked`. Resolve
dependency changes deliberately with `make lock` and commit the resulting
`uv.lock`. `make dist-check` builds and audits the public wheel and source
distribution. Never call `python` directly or `pip install` into the
environment. Put throwaway scripts in temporary files and run them through
`uv`.

Search with `rg`, never `find ... -exec grep`. Use edits that fail on missing
anchors and assert anchors in scripted edits. Do not use heredocs. Run separate
shell commands as separate calls instead of chaining them.

## Writing comments and docs (this repo is public)

Comments, docstrings and docs are public. Write for unfamiliar readers in an
impersonal, dateless OSS style. Describe the runtime or host, never "this box",
"my machine", "the rig" or a named development computer. Keep measurement
numbers while removing diary dates, first person and authority-by-person such
as "owner decision". Explain failure modes directly instead of referring to
private version history. Reserve capitals for identifiers and acronyms, using
plain words for emphasis. Avoid em dashes, antithetical definitions, patterned
sentence pairs and stock phrases such as "front door", "heart of", "by
construction", "first-class" and "the tool you reach for". State each
mechanism once. Leave program output, error messages, logs and calibration
prompt strings verbatim.

## Invariants (do not break these)

These properties hold unless a measurement proves a constraint obsolete.

**Pipeline and artifacts**
- Resolve each tensor name to a role once in the inventory. Later phases use
  typed fields without reparsing names. Shared and Qwen-style rules live in
  `inventory.roles`. Model naming contracts belong in subpackages such as
  `inventory.deepseek_v4.roles`.
- Each phase reads and writes content-hashed, versioned artifacts through the
  single contract in `core/artifact.py`. Unknown kinds, major versions and
  required features fail closed, as do missing or mismatched files.
- GGUF recipe import and probe/optimizer output converge on `package_plan`, the
  writer-facing allocation artifact. The writer follows the plan, and the
  manifest records resolved tensor decisions. Recipe paths never emit
  `optimizer_decision`. That artifact means a probe/optimizer path made the
  allocation.
- Keep build entry points in model-specific `moespresso.package` subpackages.
  GGUF-recipe builders include `package.deepseek_v4.kquant_package` and
  `package.qwen.kquant_package`, with mappings beside them in
  `package.deepseek_v4.recipe` and `package.qwen.recipe`.
  `package.kquant_recipe` supplies shared GGUF parsing and fit checks.
  `package.qwen4.iqk_convert` prepares package-ready IQ_K rows for
  `package.qwen4.iqk_package`. `package.deepseek_v4.iqk_package` assembles
  converted routed-expert IQ_K artifacts under an allocation without reading a
  GGUF recipe. `package.deepseek_v4.iqk_relayout` moves built routed bundles to
  the decode-kernel wire, while `package.iqk_format` and
  `package.iqk_relayout` define shared member and layout contracts.
- Drafter construction also belongs in `moespresso.package`.
  `package.deepseek_v4.dspark_sidecar`, `mtp_sidecar` and `dflash_sidecar`
  write sidecars. `dspark_bundle` attaches a built DSpark sidecar as a declared
  optional drafter component. Keep conversion orchestration out of `runtime/`.
  Tokenizer copying, vendored chat templates and generated jang sidecars are
  likewise package construction. Runtime loads the resulting files.
- Put model-specific correctness gates and replay/debug tools under subpackages
  such as `moespresso.correctness.deepseek_v4`. The `correctness` root holds
  shared ladder, golden, reconstruction and reference-codec helpers. Model
  probe codecs, source loaders and evidence builders belong under
  `moespresso.probe.deepseek_v4` or its family equivalent. The `probe` root
  holds shared calibration, quality, roundtrip, GGUF parsing and weight IO.
  Model source validators and naming contracts belong under
  `moespresso.inventory.deepseek_v4.static`,
  `moespresso.inventory.deepseek_v4.roles` or corresponding family packages.
  The `inventory` root holds header scanning, architecture profiles and
  shared/Qwen-style resolver helpers.
- Model runtime graph adapters, cache contracts and served-path probes belong
  under packages such as `moespresso.runtime.deepseek_v4`. The `runtime` root
  holds shared serving, HTTP, generation, verification, cache policy and generic
  streaming. Runtime reads packages and manifests without importing probe,
  optimizer or recipe-reader internals.
- `optimized_kernels_expected` defaults to false in manifest metadata. A build
  must explicitly promote it, and runtime fast paths still check tensor formats
  and shapes. Package-plan force overrides act only at build time, offer dry-run
  previews, record forced decisions in the manifest and fail closed on unknown
  formats or unmatched patterns unless explicitly allowed. A calibrated probe
  requires calibration. The uniform fallback needs explicit opt-in and cannot
  pass as calibrated.

**Runtime**
- Build from the package manifest's declared facts. Loading never reparses
  source files or guesses conventions. Render each request once in
  `http.render_prompt` and pass pre-rendered text to generation so KV-cache
  identity stays stable. SHA-256 and manifest verification remain in the
  separate `moespresso-verify` gate, outside the load and serve hot paths.
- Consult in-memory KV and prefix state first. Plain and speculative producers
  use separate rails. A speculative rail identifies cache schema, numeric
  producer lattice, drafter family, sidecar artifact and resolved schedule.
  Resumable DSpark entries pair target cache and state capsule at the same
  public token frontier. Lookup probes both rails without moving entries.
  Publish committed public state only, excluding rejected proposal rows and
  transient verify frontiers. Speculative paths lacking the complete state
  protocol use a fresh per-request cache and bypass both tiers.
- Disk KV restores a target only from exact token-prefix checkpoints at
  256-aligned frontiers. Any mismatch falls back to cold serving. The read path
  requires recorded evidence that aligned saves round-trip bit-identically for
  hybrid KV and recurrent-state caches. Serving enables the store by default
  under a per-package user-cache root, with an LRU byte budget and write-depth
  cap for shallow shared prefixes. Deep cumulative snapshots add write traffic
  without cross-session value. `MOESPRESSO_DISK_KV=off` disables the store. An
  unavailable default store degrades to memory-only serving, while an explicitly
  requested store refuses startup.
- Disk checkpoints are written during prefill only at proven live frontiers.
  Token accounting proposes a frontier, and every positional cache must report
  that exact offset independently before a write. Writes are blocking and
  atomic. A hard failure logs once and disables writes for that request.
  Restore validation failures quarantine the payload. Measure and log the
  checkpoint's TTFT cost.
- DSpark drafter-state companions form a separate disk persistence class bound
  to validated targets, with their own schema version, index, payload tree and
  quarantine tree. They never replace a target. At allowlisted aligned
  frontiers, paired prefill writes require exact, independent target-cache and
  drafter-state offsets. Commit target first, then companion. Restore target
  first, selecting a companion only for its exact identity and producer rail.
  Missing, unreadable, invalid or incompatible companions are skipped or
  quarantined separately. The target then serves the suffix with plain decoding.
  Companion read or write faults set their own disable flag so target
  checkpointing continues. Target eviction removes every dependent companion.
- Keep one fused routed-MoE operation per layer behind one Python dispatch
  boundary. Splitting resident and missing partial matmuls measured slower
  despite better wait counters. Dispatch count alone does not decide the path.

**Correctness**
- Correctness requires token or logit identity. Plausible text can hide wrong
  logits. Compare logits or top tokens against a reference after changing
  caching, routing, KV or artifact loading.
- Quality ladders are model-specific. DeepSeek-V4 uses Q0 through Q4 and a
  teacher-forced WikiText perplexity arm. Recipe or quantized-math changes
  require Q1, Q2, Q3, Q4 and perplexity. That arm alone caught a numeric blowup
  while every numbered gate remained finite. See `docs/deepseek_v4_quality.md`
  for change-specific gates. Other families need their own quality gates.
- Check the upstream release boundary before adding DS4-derived expected
  continuations, official answers or top-logprob arrays. Keep private oracle
  material outside the committed tree.

## Working method

- Measure or revert performance changes against a real numeric threshold in a
  same-artifact A/B. Below threshold, revert and briefly record why. Verify
  that both arms actually run different paths before trusting a null result.
  Judge the change with an independent metric.
- Set `MOESPRESSO_DISK_KV=off` for benchmarks unless measuring that tier.
  First-request writes inflate TTFT while later requests skip writes or restore
  checkpoints. Store state persists across processes and once inflated a
  recorded decode level by ten percent.
- Pin `MOESPRESSO_DS4_DRAFTER` identically in both arms unless measuring
  speculation, and record which arm drafted. Automatic selection depends on
  usable wired capacity and can differ across runs of the same code. The
  in-memory speculative producer rail also carries request-to-request state
  after disk KV is disabled. The startup line identifies an unpinned arm's
  resolved drafter state.
- Rerun the full builder checklist when turning a probe or one-off script into
  a builder. A shortcut previously omitted the cold-start hotlist. For shared
  mutable state such as streaming expert pools, reason through failure modes,
  test under contention, then publish while holding every required lock and
  only after all loads complete.
- Run `make lint` and `make test` before declaring any change done, including
  conversion, package writing, runtime serving, MLX/Jang/mlx-kquant/mlx-iqk
  paths and model-specific quality gates.
- Run `make roadtest` for cache, checkpoint and restart-resume changes. This
  opt-in, GPU-bound, multi-hour served-package soak is outside `make test`.
  Its fixed protocol precedes optional context growth and can exceed 50k tokens.
  `--target-tokens` stops only the extension phase, whose default target is
  110k tokens. It does not bound total context or runtime. The `agentlib/`
  runner checks cache events, disk checkpoints and restart resume across turns,
  beyond unit-test coverage. Review the full command and expected duration
  before starting it.
- Run `make dist-check` for release-facing changes. Wheel and source
  distributions must exclude `specs_archive/`, all `*/private/` fixture trees,
  environment files, caches, bytecode and machine-local paths. They must carry
  `LICENSE-MIT`, `LICENSE-APACHE-2.0`, `THIRD-PARTY-NOTICES` and the expected
  DeepSeek-V4, Ornith and technical Qwen architecture surfaces. The audit pins
  wheel `License-File` metadata to exactly those three names and compares each
  file byte for byte with the repository copy. The audit does not assess notice
  content. Update `THIRD-PARTY-NOTICES` by hand when vendoring code.

## Where to start

1. `README.md` for what the project is and how to run it.
2. `DEVGUIDE.md` for the architecture and the source map.
3. The `docs/` file for the subsystem you are touching.
4. The tests for that subsystem, which specify its behavior.
