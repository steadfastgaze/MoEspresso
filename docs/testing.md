# Native validation

`make help` lists the maintained commands. The check targets require `make`;
opt-in test gates and `make dist-check` also require ripgrep (`rg`). Cargo
commands use the committed lockfile. Run on a supported macOS Metal host with
the Rust toolchain, Clippy and rustfmt installed. Compile-only checks do not
execute Metal kernels.

## Regular checks

```sh
make lint
make dead-code
make test
```

`make lint` runs `cargo fmt --all -- --check` and
`cargo clippy --locked --all-targets -- -D warnings`. Clippy warnings fail the
gate. `make dead-code` checks the library and executable with compiler warnings
denied. These checks catch compiler-visible unused items, not every unused
public API or string-selected Metal kernel. Review callers and kernel bindings
before deleting those paths. `make test` runs regular library, executable,
integration tests, including synthetic Metal tests. Tests run
serially within each target.
Ignored package, external-oracle and acceptance tests are separate opt-in gates.
Their wrapper first verifies that the exact ignored test exists, so a renamed
or unavailable test cannot report a successful zero-test run.

The ignored-test wrapper owns each Cargo process group during test discovery
and execution. Interrupts reach only that group, and cleanup waits for owned
descendants before returning. CPU process regressions cover exact-name
preflight, signal forwarding and escalation without model weights:

```sh
cargo test --locked --test test_ignored
```

## Real-package checks

Set `MOESPRESSO_TEST_PACKAGE` to an existing Qwen3.8-Flash-Next IQ_K package.
Reserve the Metal device exclusively before running these commands, including
the ordinary synthetic Metal suite when another inference process is active.
Do not run GPU make targets concurrently in separate processes.

```sh
MOESPRESSO_TEST_PACKAGE=/path/to/package make test-model
MOESPRESSO_TEST_PACKAGE=/path/to/package make test-adaptive
MOESPRESSO_TEST_PACKAGE=/path/to/package make test-conversation
MOESPRESSO_TEST_PACKAGE=/path/to/package make roadtest
MOESPRESSO_TEST_PACKAGE=/path/to/package make test-opencode
```

`make test-model` selects tests by their exact names. They check allocated
and reusable IQ_K row reads, prefill and decode read-ahead EOF recovery, retained-reader and
workspace failures, overlapped expert publication, bounded handoff failure
recovery and injected Metal errors followed by a new request on the same
engine, plus exact memory/disk state restoration and cross-prefill-width
reuse. The snapshot test opens only its own isolated checkpoint directory;
ordinary serving disk KV is disabled. These tests have fixed inputs and
do not extend context or launch a performance campaign.

The narrow-prefill failure regression verifies sorted publication, unchanged
routing policy after a failed future read and successful retry. The retained
real-weight grouped MLP check compares outputs with an independent staged
reference at 17 and 65 rows, in full and bounded 93-slot residency.
It requires unchanged unbiased routes and guarded outputs.

`make test-adaptive` exercises growth beyond startup, release during generation,
cache preservation, committed logits and boundary failure recovery. It injects
controller observations at completed operations; it does not create system-wide
memory pressure or replace constrained-host coexistence measurements.

`make test-conversation` exercises HTTP conversation continuation with real
model execution, rendered assistant/user turns, streaming, and both cache
tiers. Deterministic time and pressure observations make adaptive growth
eligible without waiting. A follow-up must actually grow while reusing prompt
and consumed generated tokens. Trial rollback and warning-pressure release
must preserve reuse, a disk-only repeat must restore across the changed
capacity, and unrelated requests must leave useful state for both long and
sub-chunk conversations. It uses a private temporary cache root and
project-owned contexts. This does not reproduce operating-system pressure or
replace the real OpenCode gate.
The default-serving regression launches the release executable, consumes a
prompt longer than 4096 tokens, advances its actual reply into a cached
conversation, interleaves a separate request and continues that conversation.
Public health checks require ordinary scratch to be ready after the requests;
public cache accounting requires reuse of consumed generated tokens.

The request-resource regression preserves a generated prefix when ordinary
scratch allocation fails after wider prefill. Generation refuses the missing
workspace, then a new preparation boundary repairs scratch, restores ordinary
capacity and reuses the actual assistant/user follow-up. A cleanup allocation
failure preserves the primary request error and allows repair on the same
engine. A pressure release remains in effect after scratch narrows.
A separate interruption case saves a proven live wide frontier, drops the
engine, restarts at a different narrow expert capacity and resumes from the
exact disk anchor.

The full-mode disk restart case completes a sub-chunk conversation, drops its
engine and RAM state, reopens disk storage at a different expert capacity, and
requires a follow-up to reuse consumed generated tokens. It then damages the newest
checkpoint and requires fallback to an older valid generated prefix.
The retention case advances a conversation through four follow-ups, discards
RAM state, and requires independent title and summary requests to restore from
the bounded disk store. The limited-RAM case permits only one host snapshot
and requires a follow-up to select later generated state from disk instead of
replaying from the shorter RAM prompt. A damaged deeper disk entry must fall
back to that healthy RAM state.

The prefix-mode case requires only bounded prefill writes, zero writes on a
cached short follow-up, and zero writes for an independent sub-chunk prompt.
After reopening the engine and store, a conversation that branches before the
deeper anchors must still restore its earlier shared prefix. With RAM retention
below snapshot admission, repeated requests must restore only the disk anchor
and account for the re-evaluated tail without writing deeper checkpoints.

Two wide-prefill cases exercise a 2048-row workspace with real weights. One
cancels at an owner-control boundary before the chunk commits, then reuses a
previously cached conversation. The other releases expert storage during a
chunk, completes it, and requires the next request to reuse its prompt state.
Controlled observations establish lifecycle behavior without creating
system-wide memory pressure.

`make roadtest` runs the repository-owned `cache_roadtest` integration test.
It starts the native server with isolated cache and configuration directories,
then checks both the default prefix policy and explicit full persistence.
Each mode covers cold serving, memory reuse across title/summary requests,
its expected partial or complete prompt restoration after a process restart,
continuation and recovery after a streaming disconnect. The context limit is
4096 and adaptive expert capacity is disabled. It has no optional context
growth phase or long soak. The fixed sequence is designed as a short check,
not a performance campaign. Its 20-minute total timeout is a failure ceiling,
not an expected duration; server startup is limited to 180 seconds and each
HTTP request to 120 seconds.
Startup also has an inactive synthetic legacy cache over the default aggregate
allowance. The test requires its automatic reclamation; its sparse file avoids
writing gigabytes and contains no legacy conversation fixture. Regular filesystem
tests cover aggregate accounting, active legacy locks, companions/quarantine,
repeated discovery and refusal of unsafe links before cleanup.

`make test-opencode` selects `opencode_read_edit_readback`. It requires an
installed OpenCode executable and runs a fixed read, edit and read-back
task against the native server in an isolated workspace and configuration.
The client uses `run --standalone` with the configured
`native/qwen-local#acceptance` model variant. Its ordered permissions allow
only fixture reads and edits, and its provider policy allows only the loopback
native provider. The variant enables thinking with medium reasoning effort.
`MOESPRESSO_OPENCODE_BINARY` selects a specific executable instead of `opencode`
on `PATH`. The command reports the installed version. The test checks the marker
file, observed tool flow, and at least two continuations that reuse 1024 or more
prompt tokens while processing a smaller suffix. It uses adaptive serving and
the normal bounded Cache-Prior routing recipe, and checks that default logs do
not contain per-sample adaptive traces. It does not use an existing OpenCode
workspace or configuration. Both acceptance gates select a 20 GiB memory
ceiling. The cache/restart gate keeps capacity static within each server process
and uses unbiased routing to isolate cache lifecycle behavior. Restart admission
may select a different capacity or prefill width. Neither creates external memory pressure.

The ignored-test wrapper selects the acceptance target in Cargo's release
profile. Cargo supplies the matching native executable path, including when a
custom target directory is configured. Direct Cargo invocation uses the
executable from its selected profile. Set `MOESPRESSO_TEST_BINARY` only to test
a deliberately selected alternate native binary. The OpenCode workflow has a
15-minute total timeout, a 14-minute client timeout and the same 180-second
server-startup ceiling. These are failure ceilings for startup or stalled
inference, not the intended runtime of the fixed client task. Initial
compilation and model loading affect total wall time; the commands do not
promise a model-independent duration.

Acceptance runs print the selected binary, package artifact ID and resolved
expert and prefill capacities; the OpenCode check also prints its executable
and version. Temporary cache data and logs are removed after success. If a run
fails, its private temporary directory is retained and its path
is printed. That directory contains server logs and size-limited client output
useful for diagnosis. A missing package or executable is a failure, not a skipped
success.

## Local whole-model quality regression

These gates run directly against the native engine and package tokenizer. They
require no API, Python, downloaded question set or external service. Reserve
exclusive GPU ownership. They load real weights and remain outside `make test`.
The scoring and output-grading unit tests run in the regular suite.

First select a build whose behavior has been reviewed and record a baseline
outside the source tree:

```sh
MOESPRESSO_TEST_PACKAGE=/path/to/package MOESPRESSO_QUALITY_BASELINE=/path/to/baseline make quality-record
```

Then run the candidate build against those unchanged files:

```sh
MOESPRESSO_TEST_PACKAGE=/path/to/package MOESPRESSO_QUALITY_BASELINE=/path/to/baseline make test-quality
```

The short gate uses three project-owned cases: literal fact retrieval, simple
arithmetic and a declared inventory tool. It checks finite full-vocabulary
logits while teacher-forcing known continuations, then independently generates
greedy answers and a structured tool call. Answers must match their expected
values; the tool name and parsed arguments must match exactly. Reaching the
generation limit is a failure, even if an answer appears in partial output.
The gate uses release-profile code, 512-row prefill by default, a 2048-token context and
at most 96 generated tokens per case. Initial compilation and package loading
affect duration; there is no soak or automatic context extension.

The default is full residency at 512 experts with unbiased routing. On a
constrained host, set `MOESPRESSO_QUALITY_CAPACITY=128` (or another explicit
capacity between 64 and 511) for Cache-Prior 2/2 decode and unbiased prefill.
Keep this value identical for recording and comparison. Capacity and the fixed
case order matter because bounded expert residency carries across cases. No
adaptive resizing, serving prompt cache or disk cache participates.

The optional long gate traverses a project-owned maintenance ledger to 10240
tokens. It scores the next token at frontiers 2047/2048/2049, around the sparse
attention transition, and 8447/8448/8449, around the first packed KVarN tile,
then at 10240. These are teacher-forced state-boundary checks, not a long-context
reasoning benchmark. Record and compare it separately:

```sh
MOESPRESSO_TEST_PACKAGE=/path/to/package MOESPRESSO_QUALITY_BASELINE=/path/to/baseline make quality-record-long
MOESPRESSO_TEST_PACKAGE=/path/to/package MOESPRESSO_QUALITY_BASELINE=/path/to/baseline make test-quality-long
```

For a supported 4096-row prefill schedule, explicitly select the engine
workspace and chunk width with `MOESPRESSO_QUALITY_PREFILL_ROWS=4096`. Compare
against independently recorded 512-row references by also setting
`MOESPRESSO_QUALITY_REFERENCE_PREFILL_ROWS=512`. The declaration must match the
stored recipe's width; package, capacity, routing, context, token sequences,
case order and scored frontiers must still match exactly. Without the
declaration, different widths fail the recipe check. Recording rejects a
reference-width declaration and never overwrites existing baselines.

Use a separately reviewed reference build to record each 512-row suite:

```sh
MOESPRESSO_TEST_PACKAGE=/path/to/package MOESPRESSO_QUALITY_BASELINE=/path/to/reference MOESPRESSO_QUALITY_PREFILL_ROWS=512 make quality-record
MOESPRESSO_TEST_PACKAGE=/path/to/package MOESPRESSO_QUALITY_BASELINE=/path/to/reference MOESPRESSO_QUALITY_PREFILL_ROWS=512 make quality-record-long
```

Then select the candidate build and compare without changing those files:

```sh
MOESPRESSO_TEST_PACKAGE=/path/to/package MOESPRESSO_QUALITY_BASELINE=/path/to/reference MOESPRESSO_QUALITY_PREFILL_ROWS=4096 MOESPRESSO_QUALITY_REFERENCE_PREFILL_ROWS=512 make test-quality
MOESPRESSO_TEST_PACKAGE=/path/to/package MOESPRESSO_QUALITY_BASELINE=/path/to/reference MOESPRESSO_QUALITY_PREFILL_ROWS=4096 MOESPRESSO_QUALITY_REFERENCE_PREFILL_ROWS=512 make test-quality-long
```

Pin `MOESPRESSO_QUALITY_CAPACITY` identically in recording and comparison when
using bounded residency. Both `short/recipe.json` and `short/logits.bf16le`, or
their `long/` equivalents, must already exist for the selected gate. The short
prompts exercise a 4096-row workspace with shorter actual chunks. The long
case retains frontiers 2047/2048/2049 and 8447/8448/8449, then 10240; its span
from 2049 to 8447 executes a full 4096-row chunk. Selecting this quality width
does not change automatic serving's 2048-row preference.

Each suite stores a small recipe and raw BF16 rows in its own `short` or `long`
subdirectory. Missing or incompatible baselines fail before model loading.
Recording never overwrites a suite. The recipe pins package identity, token
sequences, case order, capacity, routing and prefill schedule; it has no paths,
machine identifiers or executable digests. Generated baselines stay outside
the committed repository. Capturing and comparing the same revision validates
the mechanism and repeatability, not independent model correctness. A baseline
from a qualified earlier revision can detect subsequent regression.

Reports include per-case mean negative log likelihood (NLL), maximum per-row
logit RMS difference, mean NLL increase, largest individual-token NLL increase
and changed greedy selections. Numerical thresholds default to zero: a changed
native path requires review, but a difference does not prove worse model
quality. For an explicitly reviewed numerical change, the two nonnegative
finite overrides are `MOESPRESSO_QUALITY_LOGIT_RMS` and
`MOESPRESSO_QUALITY_NLL_INCREASE`; the latter limits mean increase separately
for each case, not one pooled score. Finite logits and answer/tool checks always
remain mandatory. Do not raise thresholds or regenerate baselines merely to
obtain a pass. These small cases complement cache exactness tests and broader
quality evaluation; they do not establish general model quality.

Changed greedy selections are diagnostic counts and do not require identity
with the reference. Generated answer/tool checks evaluate the project-owned
tasks independently. Cross-width comparisons retain the same numerical limits;
the width declaration never adjusts tolerances or refreshes a baseline.

## External numerical references

```sh
MLP_SIGMOID_BF16_ORACLE=/path/to/bf16-table.bin IQK_SIGMOID_ORACLE=/path/to/fp16-table.bin make test-oracles
```

The tables contain 65,536 little-endian 16-bit outputs, one for each input bit
pattern. The BF16 check dispatches the native Metal sigmoid kernel. The FP16
check compares the CPU-generated expert lookup table. Both preserve their exact
byte comparisons and fail when a required table is absent or malformed. Keep
external oracle captures in private research storage.

## Source distribution

```sh
make dist-check
```

The source-only gate assembles files from the declared source directories
through an explicit path and extension allowlist. It rejects private paths,
links, non-source files and machine-local paths, requires the lockfile and
license/notice files, checks those files byte for byte after extraction, then
rebuilds all Cargo targets from the extracted archive with `--locked --offline`
and an independent target directory. Fetch the locked dependencies before this
offline check. It writes the successful archive to
`target/dist/moespresso-source.tar` and removes its temporary extraction/build
directory. It does not run GPU tests or construct a binary notice bundle.

Follow [distribution notices](distribution.md) for dependency and Rust toolchain
license texts required in binary releases. Review private data and attribution
content before publication; a path audit does not establish their provenance.
