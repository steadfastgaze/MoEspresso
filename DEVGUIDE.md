# Developer guide

This guide maps the source tree to the package lifecycle. Read `AGENTS.md`
before changing code. The [documentation index](docs/README.md) points to the
subsystem contracts and model guides.

DeepSeek-V4-Flash and Ornith are the public product models. Ornith uses the
`qwen` architecture adapter. The separate `qwen4` adapter supports
Qwen3.8-Flash-Next with composite recurrent state, KVarN sparse attention,
and SSD-backed PLE tables. Technical architecture support does not declare a
published model package. See [Qwen4 serving](docs/qwen4.md),
[Cache-Prior routing](docs/cache_prior.md), and the opt-in
[Qwen4 MTP path](docs/qwen4_mtp.md). Ordinary Qwen4 serving leaves MTP off.

## Package lifecycle

Model-specific builders turn a GGUF recipe or converted expert artifacts with
an allocation into a `package_plan`. The shared writer consumes that plan and
records the resolved tensor decisions in the package manifest. Calibrated
probe and optimizer workflows retain their input identities. Runtime loading
starts from the manifest and never imports probe, optimizer, recipe-reader, or
conversion-stage artifact internals. It does not branch on the planning route
that produced the package.

`src/moespresso/core/artifact.py` owns content hashes, version checks, and
fail-closed artifact reads. The [artifact contract](docs/artifact_contract.md)
defines those rules, and the [conformance matrix](docs/artifact_conformance_matrix.md)
shows the fields each artifact carries. The package itself consists of
safetensors shards and `package_manifest.json`, with architecture facts,
package-plan provenance, declared file hashes, tokenizer and rendering
identity, required operations, and per-tensor formats.
Its eight storage formats are `affine`, `mxfp4`, `mxfp8`, `kquant`, `iqk`,
`fp16`, `f32_passthrough`, and `raw_dtype_passthrough`. Routed-expert bundles
keep each streamed row contiguous. See the [package format](docs/package_format.md).

## Source map

| Source | Responsibility | Guide |
|---|---|---|
| `core/` | Shared artifact contract and validation. | [Artifact contract](docs/artifact_contract.md) |
| `inventory/` | Header-only source inventory, role resolution, family profiles, and model-specific naming checks. | [Source inventory](docs/source_inventory.md) |
| `probe/` | Calibration, weight and GGUF readers, measured evidence, and allocation inputs. | [Probe evidence](docs/probe_evidence.md), [optimizer decisions](docs/optimizer_decision.md) |
| `package/` | Plans, deterministic shard writing, manifests, compatibility files, codec contracts, and model-specific builders. | [Package format](docs/package_format.md) |
| `correctness/` | Standalone package checks, public and private fixture boundaries, and model-specific quality gates. | [Correctness ladder](docs/correctness_ladder.md) |
| `toolcalls/` | Shared pure-stdlib dialect parsers, serializers, and bounded repair for serving and agent clients. | [Tool calls](docs/tool_calls.md) |
| `runtime/` | Manifest-driven model loading, generation, HTTP and SSE serving, verification, caching, and model adapters. | [Resident runtime](docs/runtime_resident.md), [disk KV](docs/disk_kv.md) |
| `runtime/` pooled routed modules | Persistent expert slots and dispatch for K-quant, MXFP4, and IQ_K. Full capacity serves resident rows, while smaller pools read missing rows from disk. | [SSD streaming](docs/ssd_streaming.md) |
| `agentlib/` | HTTP agent loop, SSE client, sandboxed tools, and the served road-test harness. It reads a declared `agentic_profile.json` when present without importing runtime internals. | [Package format](docs/package_format.md) |

Model-specific code lives below `inventory/`, `probe/`, `package/`,
`correctness/`, and `runtime/`. DeepSeek-V4 speculative decoding uses the
DSpark and DFlash sidecar builders in `package/deepseek_v4/` and the draft,
verify, rollback, and selection code in `runtime/deepseek_v4/`. DSpark state
can accompany a target disk checkpoint while the target remains authoritative.
The [speculative decoding guide](docs/speculative_decoding.md) covers its
admission and cache contracts. DeepSeek MTP has no installed entry point or
serving selector.

`runtime/pooled_moe.py` shares routed scheduling across models, with
`runtime/pooled_decode_session.py` as the request owner. Full residency
changes pool capacity under that same owner.
`runtime/pooled_moe_blocks.py` adapts DeepSeek and Ornith math, while Qwen4
supplies its own reduction and padded projections to the same scheduler.
The Qwen4 bounded full512 default uses cache-conditioned decode routing,
with original routing during prefill and full residency.
`mlx-kquant` and `mlx-iqk` provide pinned quantized kernels, and
`runtime/expert_slot_pool.py` owns routed slot storage. The native
MTLSharedEvent gate under `native/gate/` is built by `uv sync --locked` and
checked for capability at runtime. Unsupported native-gate capabilities use
the ring fallback, and imports never compile the extension. See
[native setup](native/README.md).

`tests/` mirrors the implementation and specifies subsystem behavior. For
performance work, follow the [optimization methodology](docs/optimization_methodology.md).

## Serving and diagnostics

`moespresso serve` exposes OpenAI-compatible `POST /v1/chat/completions` and
`GET /health`, with SSE streaming and served tool calls. It renders each prompt
once and warms generation before reporting readiness. The default context
limit is 128K or the package limit, whichever is smaller. DeepSeek-V4 may
choose a smaller default to preserve its minimum expert pool.
`--max-context-tokens` can select a positive limit up to the architecture
limit. Prefix reuse checks memory first, then the default-on per-package disk
KV tier. `MOESPRESSO_DISK_KV=off` disables that tier. A supported bundled
drafter engages when its files, resident experts, and wired-memory capacity
qualify. Other requests use plain decoding. `MOESPRESSO_DS4_DRAFTER=off`
disables drafting. See
[resident serving](docs/runtime_resident.md), [disk KV](docs/disk_kv.md), and
[speculative decoding](docs/speculative_decoding.md) for the full rules.

`moespresso verify` checks package integrity separately from serving.
`moespresso speed` summarizes server-reported decode speed from an existing
server, while `moespresso completions-api-timing` uses client timestamps and
local tokenization against an existing streaming API. See
[diagnostics](docs/diagnostics.md) and
[API timing](docs/completions_api_timing.md). Internal Metal-trace and process
resource readers live in `runtime/decode_trace.py`,
`runtime/decode_trace_detail.py`, `runtime/diagnostic_environment.py`, and
`runtime/process_resources.py`. They have no installed profiling command.

## Commands

The installed entry points are declared in `pyproject.toml`.

| Task | Commands |
|---|---|
| Generate, serve, and verify | `moespresso generate`, `moespresso serve`, `moespresso verify`. The legacy aliases are `moespresso-generate`, `moespresso-serve`, and `moespresso-verify`. |
| Inspect and measure | `moespresso-hf-inspect` (`hf-model-inspect` alias), `moespresso speed`, `moespresso completions-api-timing`, `moespresso-ds4-speed-stats`. |
| Build DeepSeek-V4 packages | `moespresso-ds4-kquant-package`, `moespresso-ds4-iqk-package`, `moespresso-ds4-iqk-relayout`, `moespresso-ds4-iqk-reap`. Relayout and REAP retain converted bytes. |
| Build Qwen-family packages | `moespresso-qwen-kquant-package` for the Ornith architecture, plus `moespresso-qwen4-iqk-convert` and `moespresso-qwen4-iqk-package` for Qwen4. `moespresso-qwen4-mtp` is an explicit full-resident experiment. |
| Build and assess DeepSeek-V4 drafters | `moespresso-ds4-dspark-sidecar`, `moespresso-ds4-dflash-sidecar`, `moespresso-ds4-dspark-bundle`, `moespresso-ds4-dspark-replay`, `moespresso-ds4-spec-battery`. |
| Run quality gates | `moespresso-ds4-quality`, `moespresso-ds4-q4`, `moespresso-ds4-wikitext-ppl`, `moespresso-ds4-q1-validate`, `moespresso-ornith-gate`, `moespresso-qwen35-hard-questions`. |

The [DeepSeek package recipe](docs/deepseek_v4_package_recipe.md),
[quality guide](docs/deepseek_v4_quality.md), and
[speed guide](docs/deepseek_v4_speed.md) give the model-specific commands and
evidence. The DeepSeek-V4 quality gates run manually against a package:

```sh
uv run --locked moespresso-ds4-quality q1 --package <package>
uv run --locked moespresso-ds4-quality q2 --package <package>
uv run --locked moespresso-ds4-quality q3 --package <package>
```

`moespresso-ornith-gate` runs the public agentic-coding and long-context
families with:

```sh
uv run --locked moespresso-ornith-gate <package> --families agentic_coding,long_context
```

The full `hard_reasoning` family requires ignored private Ornith questions,
keys, and verification programs. Public source-release tests inject synthetic
fixtures. DeepSeek-V4 fixtures published by antirez/ds4 remain committed,
while provider captures stay in ignored private fixture directories regardless
of gate number. Private payloads never enter release artifacts.

## Build and release checks

`make install` syncs runtime and development dependencies. `make lock`
deliberately re-resolves them, while `make lock-check` checks the lock without
changing it. `make fmt` formats the tree. `make lint` and `make test` validate
it with `uv run --locked`. Real-model tests and model-specific quality gates are
opt-in. Changes to runtime math or package formats must clear the affected
family's gates, in addition to the full test suite.

`make roadtest` runs the multi-hour served certification soak for cache,
checkpoint, and restart-resume changes. Its fixed protocol includes restart
replay, interleaved and delegated sessions, and can exceed 50K tokens before
optional context growth. The default extension target is 110K tokens.
`--target-tokens` bounds only that optional phase. Review the full command and
expected duration before starting it.

For release-facing changes, `make dist-check` builds and audits the wheel and
source distribution for private-file exclusion, required licenses, and product
surfaces. Keep the release artifacts free of private fixtures and local files.
