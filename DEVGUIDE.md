# Developer guide

This is the map from the source tree to the design. See also `AGENT.md`.
For each subsystem in depth, see the matching file in `docs/`.

Public product support covers DeepSeek-V4-Flash and Ornith. Modules and entry
points named `qwen` implement the Qwen architecture that Ornith uses and retain
useful engineering harnesses. They are architectural namespaces rather than a
third public product.

## The lifecycle

MoEspresso has two planning routes that converge before writing a package.
The research route measures a source model and optimizes an allocation. The
recipe route imports an existing GGUF allocation. Both must emit the same
internal `package_plan`, and the writer/runtime must not care which route
produced it.

```
inventory  ->  probe  ->  optimize  --\
                                      package_plan  ->  package  ->  runtime
GGUF recipe import  -----------------/
```

| Phase | Reads | Writes | What it does |
|---|---|---|---|
| inventory | source model headers | `source_inventory` | classify every tensor, resolve its role and imatrix key once |
| probe | `source_inventory` + weights + calibration | `probe_evidence` | measure activation-weighted reconstruction quality per (bits, group-size) |
| optimize | `probe_evidence` + constraints | `optimizer_decision` | allocate bits per tensor under a size/quality/tail constraint |
| recipe import | GGUF metadata + source inventory | recipe allocation | copy an external tensor-by-tensor format recipe without copying GGUF weights |
| package plan | `optimizer_decision` or recipe allocation | `package_plan` | normalize allocation, provenance, and explicit force overrides into the writer IR |
| package | `package_plan` + weights | `package_manifest` | quantize, write shards, emit the package's self-description |
| runtime | a package | (generation) | build the model from the manifest, verify, serve |

Every artifact carries a content-hashed id and a fail-closed version. The
contract is one file: `src/moespresso/core/artifact.py`. It registers seven
kinds: the five pipeline kinds above plus two standalone correctness-ladder
kinds (`architecture_profile`, `correctness_evidence`). See
`docs/artifact_contract.md`; `docs/artifact_conformance_matrix.md` maps what
each produced artifact actually carries.

## Module map (`src/moespresso/`)

| Package | Modules | Role | Reference |
|---|---|---|---|
| `core/` | `artifact.py` | the base contract: keys, canonicalization, content hash, fail-closed versioning | `docs/artifact_contract.md` |
| `inventory/` | `roles.py`, `build.py`, `architecture_profile.py`, `safetensors_header.py`, `hf_inspect.py`, `deepseek_v4/` | header-only inventory; shared/Qwen-style name-to-role resolver helpers; model-specific naming contracts and static source validators; the family correctness contract | `docs/source_inventory.md` |
| `probe/` | `build.py`, `calibration.py`, `quality.py`, `roundtrip.py`, `weight_io.py`, `gguf_parse.py`, `deepseek_v4/` | streamed quality measurement; the GGUF/legacy imatrix calibration provider; model-specific probe adapters | `docs/probe_evidence.md` |
| `optimize/` | `allocate.py`, `decide.py`, `aggregate.py`, `health.py`, `monotone.py`, `sizes.py`, `affine_elasticity.py` | the bit-allocation core, the decision artifact, the allocation health gate | `docs/optimizer_decision.md` |
| `package/` | `convert.py`, `plan.py`, `manifest.py`, `write.py`, `bundle.py`, `hotlist.py`, `tokenizer.py`, `templates.py`, `templates/` (vendored chat templates), `sidecars.py`, `agentic_profile.py`, `constants.py`, `tq.py`, `kquant_format.py`, `kquant_recipe.py`, `kquant_backend.py`, `kquant_bundle.py`, `kquant_cache.py`, `kquant_gguf.py`, `deepseek_v4/`, `qwen/` | build-time orchestration, the common package-plan IR, the byte-deterministic writer, the manifest, tokenizer/template packaging, sidecar and agentic-profile generation, shared MJTQ/K-quant format contracts, shared GGUF K-quant parsing, encode cache, and model-specific GGUF recipe/package builders | `docs/package_format.md` |
| `correctness/` | `ladder.py`, `gate.py`, `goldens.py`, `reconstruct.py`, `tq_reference.py`, `environment.py`, `deepseek_v4/`, `qwen35/`, `ornith/` | the general correctness ladder, the convert-time gate, environment gating for measured runs, and model-specific quality gates and debug tools | `docs/correctness_ladder.md` |
| `runtime/` | `serve.py`, `build.py`, `http.py`, `chat_stream.py`, `generation.py`, `verify.py`, `thinking.py`, `kv_policy.py`, `prefix_cache.py`, `disk_kv.py`, `kquant_install.py`, `owned_switchglu.py`, `deepseek_v4/`, `qwen/` | serve a package from its manifest; one-shot and HTTP entry points with SSE streaming; in-memory KV and prefix reuse; declared-context-limit refusal; the opt-in disk KV checkpoint tier; K-quant module installation; model-specific runtime adapters, kernels, and probes | `docs/runtime_resident.md`, `docs/disk_kv.md` |
| `runtime/` (streaming) | `ssd_streaming_build.py`, `streaming_capacity.py`, `expert_index.py`, `expert_loader.py`, `expert_pool.py`, `expert_slot_pool.py`, `expert_locality.py`, `pooled_switchglu.py`, `routed_decode_kernel.py`, `gather_tq_split_norms.py`, `pread_into.py`, `native_gate.py`, `streaming_run_lock.py` | stream routed experts from disk within a memory budget | `docs/ssd_streaming.md` |
| `agentlib/` | `client.py`, `conversation.py`, `sse.py`, `toolcalls.py`, `repair.py`, `qwenxml.py`, `dsml.py`, `envelope.py`, `loop_policy.py`, `profile.py`, `sandbox.py`, `subagent.py`, `tools.py`, `execution.py`, `roadtest/`, `dialect_study/` | an agent loop over the served HTTP surface: SSE consumption, tool-call dialect parsing and repair, loop policy, agentic-profile resolution, sandboxed tool execution, and the served road-test harness. It speaks to the server over HTTP and reads the package's `agentic_profile.json`; it imports no runtime internals | `docs/package_format.md` (agentic profile) |

Native Metal primitives live under `native/` (`gate/` for the MTLSharedEvent
decode gate, `ds4_moe/` for the DeepSeek-V4 MoE kernels) and are built
separately via `native/build.sh`; the runtime loads them if present and falls
back transparently if not.

`docs/mlx_limitations.md` records the measured limits and strengths of the MLX
platform this runtime is built on. `docs/optimization_methodology.md` records
the measurement discipline for serving-optimization work.

## The package format

A package is a directory of safetensors shards plus a
`package_manifest.json`. The manifest is the contract: architecture facts copied
from the source config, the package-plan provenance, the on-disk format per
tensor with its parameters, every file's path/size/sha256, the required backend
operations, and the tokenizer/rendering identity. Eight on-disk tensor formats
exist: `tq`, `affine`, `mxfp4`, `mxfp8`, `kquant`, and the passthrough trio
`fp16`, `f32_passthrough`, `raw_dtype_passthrough`. Routed experts are stored as
a per-layer bundle so a streamed expert miss costs one contiguous read. Shard
writing is byte-deterministic: identical inputs produce identical shard files
and hashes. Families with recorded agent-loop evidence also carry an
`agentic_profile.json` sidecar recorded in the manifest. See
`docs/package_format.md`.

The runtime consumes only the manifest and tensor formats. It must not import
probe/optimizer code or GGUF recipe readers, and it must not branch on whether a
package came from the probe/optimizer route or the GGUF recipe route.

## Serving surface

`moespresso-serve` binds an OpenAI-compatible `POST /v1/chat/completions` plus
`GET /health`. The endpoint supports SSE streaming (`stream: true`) with
`reasoning_content`/`content` deltas, sampling pass-through
(`top_k`, `min_p`, `presence_penalty`), refusal of unsupported
`repetition_penalty` values, and refusal of requests over the package's
declared context limit. The server warms generation before announcing
readiness. Prefix reuse is in-memory by default; setting
`MOESPRESSO_DISK_KV=frontier` adds the opt-in restart-warm disk checkpoint
tier. Details: `docs/runtime_resident.md` and `docs/disk_kv.md`.

## Entry points

Declared in `pyproject.toml`:

- `moespresso-convert`: end-to-end probe/optimizer conversion.
- `moespresso-generate`: one-shot generate from a package.
- `moespresso-serve`: OpenAI-compatible HTTP server.
- `moespresso-verify`: on-demand integrity gate (kept off the serve hot path).
- `moespresso-hf-inspect` (alias `hf-model-inspect`): remote HF model header
  inspection without downloading.
- `moespresso-ds4-kquant-package`: manual DeepSeek-V4 package builder from a GGUF K-quant recipe.
- `moespresso-qwen-kquant-package`: technical Qwen-architecture package builder
  used by the Ornith path.
- `moespresso-ds4-quality`: manual-only DeepSeek-V4 Q0/Q1/Q2/Q3 quality gates.
- `moespresso-ds4-q1-validate`: validate external DeepSeek-V4 Q1 parity evidence.
- `moespresso-ornith-gate`: manual Ornith quality gate v2 (reasoning, agentic
  coding, long-context recall).
- `moespresso-qwen35-hard-questions`: technical exact-answer comparison harness
  for the Qwen architecture used by Ornith.
- `moespresso-ds4-speed-stats`: manual DeepSeek-V4 speed snapshot.
- `moespresso-ds4-speed-primitives`, `moespresso-ds4-ratio4-attention-replay`,
  `moespresso-ds4-layer-stage-replay`, `moespresso-ds4-indexed-attention-probe`,
  `moespresso-ds4-moe-block-replay`, `moespresso-ds4-moe-graph-replay`: focused
  DS4 speed/correctness probes.

## Building and testing

`make install` syncs the complete runtime and development environment. `make
test` and `make lint` run lock-strict (`uv run --locked`), so an unlocked
dependency edit fails fast; `make lock` is the deliberate re-resolve step and
`make lock-check` is the read-only staleness gate. `make fmt` formats. `make
roadtest` runs the agentlib road-test harness against a served package. Tests
run through `python -m pytest` in that locked environment.

Run the full suite for changes touching conversion, package writing, runtime
serving, MLX/Jang/mlx-kquant paths, or model-specific quality gates. `make
dist-check` builds and audits the
wheel and source distribution for the public/private boundary, licenses, and
required product surfaces.

Heavy real-model tests are opt-in by environment variable and are intentionally
skipped by default. Model-specific quality gates are manual-only:

```
uv run --locked moespresso-ds4-quality q1 --package <package>
uv run --locked moespresso-ds4-quality q2 --package <package>
uv run --locked moespresso-ds4-quality q3 --package <package>
uv run --locked moespresso-ornith-gate <package> --families agentic_coding,long_context
```

The command above is the public Ornith gate. Adding `hard_reasoning` requires
the ignored private benchmark questions and answer key. Public source-release
tests use synthetic injected fixtures and never depend on those local files.

DeepSeek-V4 Q0/Q1 prompts and provider vectors are committed public fixtures
matching antirez/ds4. Q2 provider captures and the unpublished Ornith benchmark
questions, keys, and verification programs stay in ignored `*/private/`
directories and never enter release artifacts.

Any change to runtime math or package formats must clear the touched family's
gates before landing; `docs/correctness_ladder.md` covers the build-time rungs
and the model-specific gates that sit above them.
