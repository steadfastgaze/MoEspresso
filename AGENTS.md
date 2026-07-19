# Agent and contributor guide

Read this before changing anything. It covers how to work in this repo and the
invariants that must hold. For the architecture, see `DEVGUIDE.md` and `docs/`.

## Public product and fixture boundary

The public product models are DeepSeek-V4-Flash and Ornith. Source paths and
entry points containing `qwen` describe the Qwen architecture implementation
used by Ornith, plus engineering harnesses around that implementation. Their
presence does not add a third public product.

Committed DeepSeek-V4 Q0/Q1 prompts and provider vectors remain public so the
gates match the fixtures published by antirez/ds4. Q2 provider captures remain
under `src/moespresso/correctness/fixtures/deepseek_v4/private/`. Unpublished
Ornith benchmark questions, answer keys, and verification programs remain under
`src/moespresso/correctness/fixtures/ornith/private/`. Never copy either private
set into source, tests, logs, documentation, wheels, or source distributions.

The public Ornith gate surface covers the project-owned agentic-coding and
long-context instruments:

```
uv run --locked moespresso-ornith-gate <package> \
  --families agentic_coding,long_context
```

The full hard-reasoning gate additionally requires the ignored private Ornith
fixtures. Source-release tests must inject synthetic questions and keys instead
of reading those files.

## Tooling discipline

- **Use `uv` for everything.** `uv sync` sets up the complete runtime and
  development environment. Tests and lint run lock-strict (`uv run --locked`), so an
  unlocked dependency edit fails immediately. `make lock` is the deliberate
  re-resolve step; commit the updated `uv.lock`. `make dist-check` builds and
  audits the public wheel and source distribution. Never call `python` directly
  and never `pip install` into the environment.
- **Search with `rg`.** Do not use `find ... -exec grep`.
- **Edit with a tool that fails loudly on a missing anchor.** A scripted patch
  that silently misses its anchor and writes nothing is the most common way a
  change appears done but is not. Prefer an editor that errors on a miss; if you
  must script an edit, assert the anchor.
- **No heredocs**, and do not chain shell commands with `&&` or `;` when each is
  already a separate allowed command. Run them as separate calls.
- **Put throwaway scripts in temporary files** and run them via `uv`.

## Writing comments and docs (this repo is public)

Every comment, docstring, and doc ships to a public GitHub repo. Write them the
way a large OSS project does: about the code, impersonal, and dateless. Write for
an unfamiliar reader. These mandatory rules came from real cleanup work.

- **No machine references.** Never "this box", "on this box", "my box/machine/
  laptop/mac", "the rig", "small box", "big box", "16 GB box". The computer is
  never the subject of a sentence. Say "the runtime", "the host", "under memory
  pressure", "when the system is idle".
- **No dates or dev-diary notes.** No "NOTE (2026-06-10):", "Measured on the rig
  (date):", "(owner decision 2026-06-12)". State the finding as a present-tense
  fact and keep the measurement number; drop the date and the first person.
  ("Extending the kick to all-hit layers measured flat (5.42 vs 5.48 tok/s), so
  the gate stays miss-only.")
- **No authority-by-person.** No "owner decision", "owner priority", "product
  priority (owner)". State the decision as a fact: "Default ON; `FOO=0` is the
  kill switch."
- **No private version archaeology.** "pre-v8 gibberish", "the v4 collapse" mean
  nothing to an outside reader. Name the *failure mode* instead: "storing conv1d
  pre-transposed suppresses the coupled norm shift, so norms load ~1.0 too low
  and the model emits garbage."
- **No shouting-caps for emphasis.** Not "THIS", "ONLY", "MUST", "NOT", "GARBAGE",
  "LOUD". Caps are for real identifiers and acronyms (MLX, GPU, IO, LFU, env-var
  names) only. Use plain words or `**bold**` in docs if you must stress a point.
- **No lazy LLM-ese in docs.** Avoid em dashes, antithetical definitions,
  parallel equal-length sentence pairs, and phrases such as "front door",
  "heart of", "by construction", "first-class", or "the tool you reach for".
  Use plain declarative sentences that state the mechanism. Say a fact once.
- **Leave program output and content strings alone.** Error messages, log lines,
  and calibration prompt strings are not comments; do not de-shout or reword them
  for tone.

## Invariants (do not break these)

These are properties the system depends on. Each was learned from a real
failure; respect them or prove with a measurement that the constraint no longer
holds.

**Pipeline and artifacts**
- Resolve once. The inventory maps every tensor name to a role a single time;
  every later phase reads typed fields and never reparses names. Shared/Qwen
  rules live in `inventory.roles`; model-specific naming contracts live under
  model inventory subpackages such as `inventory.deepseek_v4.roles`.
- Every phase reads and writes a content-hashed, versioned artifact. Keep the
  contract in `core/artifact.py` as the single implementation. Fail closed on an
  unknown kind, an unknown major version, an unknown required feature, or a
  missing/mismatched file.
- Package builders use `package_plan` as the writer-facing allocation artifact.
  GGUF recipe import and probe/optimizer output both converge there. The writer
  consumes the plan; the manifest records the resolved tensor decisions.
- Build-time entry points live in `moespresso.package`: `package.convert` for
  probe/optimizer conversion and model-specific subpackages such as
  `package.deepseek_v4.kquant_package` or `package.qwen.kquant_package` for
  GGUF-recipe package builders. Model-specific GGUF recipe mapping lives beside
  those builders, for example `package.deepseek_v4.recipe` and
  `package.qwen.recipe`; `package.kquant_recipe` is the shared GGUF parser and
  fit-check helper. Do not put conversion orchestration back under `runtime/`.
- Package-owned compatibility files live in `moespresso.package`: tokenizer
  copying, vendored chat templates, and generated jang sidecars are package
  construction concerns. Runtime loads the resulting package files.
- Model-specific correctness gates and replay/debug tools live below a model
  subpackage such as `moespresso.correctness.deepseek_v4`. Keep the
  `correctness` root for shared ladder, gate, golden, reconstruction, and
  reference-codec helpers.
- Model-specific probe codecs, source loaders, and evidence builders live below
  a model subpackage such as `moespresso.probe.deepseek_v4`. Keep the `probe`
  root for shared calibration, quality, roundtrip, GGUF parsing, and weight IO.
- Model-specific source validators and naming contracts live below an inventory
  model subpackage such as `moespresso.inventory.deepseek_v4.static` and
  `moespresso.inventory.deepseek_v4.roles`. Keep the `inventory` root for shared
  header scanning, architecture profiles, and shared/Qwen-style resolver helpers.
- Recipe paths do not emit optimizer decisions. An `optimizer_decision` artifact
  means the probe/optimizer path made the allocation.
- Runtime code reads packages and manifests. It must not import probe,
  optimizer, or recipe-reader internals.
- Model-specific runtime graph adapters, cache contracts, and served-path probes
  live below a model subpackage such as `moespresso.runtime.deepseek_v4`. Keep
  the `runtime` root for shared serving, HTTP, generation, verification, cache
  policy, and generic streaming infrastructure.
- `optimized_kernels_expected` is manifest metadata, default false. Setting it
  is an explicit package-build promotion, and runtime fast paths still validate
  actual tensor formats and shapes before use.
- Package-plan force overrides are build-time tools. They fail closed on unknown
  formats and unmatched patterns unless explicitly allowed, support dry-run
  previews, and record forced decisions in the manifest.
- Calibration is a required input for a calibrated probe. The uniform fallback is
  an explicit opt-in that cannot pass as calibrated.

**Runtime**
- Serve from the manifest. Build the model from the package's own declared
  facts; never reparse source files or guess conventions at load.
- One template render per request. Multiple renders break KV-cache identity.
  `http.render_prompt` is the only render site; the generate path consumes
  pre-rendered text.
- Verification (sha256 + manifest checks) stays off the serve hot path as the
  separate `moespresso-verify` gate. Do not add it to the load path.
- KV cache and prefix reuse are in-memory first. The disk KV read path
  restores only exact token-prefix checkpoints at 256-aligned frontiers,
  fails closed to cold serving on any mismatch, and is gated by the
  recorded safety evidence that aligned saves round-trip bit-identically
  on hybrid (KV + recurrent-state) caches. Serving enables the store by
  default under a per-package root in the user cache directory with an
  LRU byte budget and a write-depth cap (checkpoints cover the shallow
  shared-prefix region; deep cumulative snapshots are write traffic with
  no cross-session value); `MOESPRESSO_DISK_KV=off` is the kill switch,
  and a default-enabled store that cannot open degrades to memory-only
  serving while an explicitly requested one refuses startup. Checkpoint
  writes happen only at proven live frontiers during prefill: token
  accounting proposes, and every positional cache must independently
  report exactly the frontier offset before a write, so an unaligned
  write is structurally impossible. Writes are blocking, atomic,
  quarantined on failure, and their TTFT cost is measured and logged,
  never hidden.
- One fused routed-MoE operation per layer, one dispatch boundary to Python.
  Never split a routed op into resident and missing partial matmuls; that
  measured slower despite better wait counters.

**Correctness**
- Correctness requires token or logit identity. Plausible text can still hide
  wrong logits. Any change to caching, routing, KV, or
  artifact loading must compare logits or top-tokens against a reference.
- Quality ladders are model-specific. DeepSeek-V4 keeps the Q0/Q1/Q2/Q3 gates;
  other model families need their own quality gates.
- DS4-derived evaluation fixtures need an explicit public/private boundary.
  Before adding expected continuations, official-answer files, or top-logprob
  arrays, check the upstream release boundary. Private oracle material belongs
  outside the committed tree.

## Working method

- Measure or revert. A performance change must clear a real numeric threshold on
  a same-artifact A/B; below it, revert with a short note on why.
- Prove an A/B's two arms actually differ before trusting a null result. A common
  trap is two arms that silently run the same code path.
- A metric used to judge a change must be independent of the change.
- When you repurpose a probe or a one-off script as a builder, re-run the full
  builder checklist; a shortcut once shipped packages missing the cold-start
  hotlist.
- For any shared-mutable-state code (the streaming expert pools), pre-reason the
  failure modes and test under contention. Publish after all loads complete and
  under all locks.
- Run `make lint` and `make test` before considering a change done, including
  for changes touching conversion, package writing, runtime serving,
  MLX/Jang/mlx-kquant paths, or model-specific quality gates.
- Run `make dist-check` for release-facing changes. The wheel and source
  distribution must exclude `specs_archive/`, every `*/private/` fixture tree,
  environment files, caches, bytecode, and machine-local paths. They must carry
  both license files and the expected DeepSeek-V4, Ornith, and technical Qwen
  architecture surfaces.

## Where to start

1. `README.md` for what the project is and how to run it.
2. `DEVGUIDE.md` for the architecture and the source map.
3. The `docs/` file for the subsystem you are touching.
4. The tests for that subsystem; they are the behavior specification.
