# Contributor guide

Read `README.md` and the applicable `docs/` files before changing a subsystem.
Prefer concrete module boundaries over speculative interfaces.

Use `name.rs` for a module and `name/` for its child modules. Prefer ordinary
`mod` declarations over `include!` or path overrides for handwritten child
modules. Keep extracted unit-test files under their owning module, with
descriptive names.

Measure completed prefill and decode with comparable artifacts, routing and
cache state. Dispatch counts alone do not establish performance.

Use Cargo for this repository. Commit Cargo.lock and run tests with --locked
after intentional dependency resolution. Use rg for searches and apply_patch
for edits. Do not push or publish without explicit authorization.

Check results against appropriate independent references. Maintain compatibility
with supported model packages, including tensor layouts and codec semantics.
Validate logits and resulting quality when changing numerical execution;
plausible text or relaxed tests do not establish correctness. Evaluate alternative
numerical implementations through measured quality and performance, without
requiring reference bit patterns. Handle cache-producer identity and compatibility
explicitly when a numerical path changes. Retain readers and allocations until
GPU completion, and publish expert rows only after writes finish. Engine state
and inference are owned by the serialized inference loop.

Keep investigation notes and generated measurements in a private research
directory outside this repository. Never commit model weights, private prompts,
generated outputs, captured tensors or credentials. Preserve licenses and source
attribution when adapting code.

Comments, docstrings and documentation are public. Write for unfamiliar readers
in an impersonal, dateless OSS style. Describe the runtime or host without
machine-local paths, development diaries, first person or authority-by-person.
Keep measurement numbers and explain failure modes directly. Reserve capitals
for identifiers and acronyms. Avoid em dashes, antithetical definitions,
patterned sentence pairs and stock phrases such as "front door", "heart of",
"by construction", "first-class" and "the tool you reach for". State each
mechanism once. Preserve program output, error messages, logs, calibration
prompt strings and third-party license text verbatim.

Run focused tests during development, then `make lint` and `make test` before
declaring a change complete. `make lint` checks formatting and runs Clippy with
warnings denied on all Cargo targets. `make test` includes synthetic Metal tests
and requires a supported macOS Metal host. Reserve exclusive GPU ownership for
`make test-model`, `make roadtest` and `make test-opencode`; select the package
explicitly with `MOESPRESSO_TEST_PACKAGE`. Never run every ignored test as a
single campaign. External sigmoid oracle checks use `make test-oracles` and
must fail if the selected oracle is absent. Run `make dist-check` for source
release changes and follow `docs/distribution.md` for binary notice bundles.
`make help` and `docs/testing.md` describe these maintained gates.
Run `make test-conversation` for request admission, prompt reuse or
adaptive/cache integration changes. It requires an explicit real package and
exclusive GPU ownership, and requires actual growth, trial rollback, pressure
release and interleaved short/long conversation reuse.

Use `make test-quality` with a reviewed local baseline for changes to model
arithmetic, execution or tokenization. It checks teacher-forced logits and
project-owned answer/tool cases without an API. `make test-quality-long` adds
optional sparse-attention and packed-KV boundary coverage; select it when
changing those paths. Both need `MOESPRESSO_TEST_PACKAGE`,
`MOESPRESSO_QUALITY_BASELINE` and exclusive GPU ownership. Record baselines
explicitly with `make quality-record` or `make quality-record-long`; never
refresh them or relax tolerances automatically to make a candidate pass. Keep
generated baselines outside the source repository.

Use short real-weight checks for kernel and storage work. Reserve long model
benchmarks for changes that require them. Keep one-off performance campaigns
in the private research directory.
