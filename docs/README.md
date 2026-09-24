# Documentation

This directory is the public reference for building, validating, and serving
MoEspresso packages. It is self-contained: public users and contributors should
not need private investigation logs to understand the supported paths.

## Model guides

### DeepSeek-V4-Flash

- [`deepseek_v4_package_recipe.md`](deepseek_v4_package_recipe.md): inputs,
  preflight, byte-faithful builds, diagnostics, cache behavior, verification,
  and serving.
- [`deepseek_v4_quality.md`](deepseek_v4_quality.md): Q0 through Q4, perplexity semantics,
  commands, acceptance evidence, and the public/private fixture boundary.
- [`deepseek_v4_speed.md`](deepseek_v4_speed.md): benchmark protocol, current
  resident and streamed records, context envelope, and known gaps.
- [`speculative_decoding.md`](speculative_decoding.md): drafter families and
  sidecars, the draft/verify loop, the adaptive scheduler, producer-scoped
  cache reuse, memory, and the correctness contract.

### Qwen4 architecture

- [`qwen4.md`](qwen4.md): Qwen3.8-Flash-Next's separate adapter, PLE storage,
  KVarN state, ordinary decoding and prompt reuse.
- [`cache_prior.md`](cache_prior.md): Cache-Prior, the dynamic REAP analogy,
  Apple's prompt-level routing, and the LFU eviction policy.
- [`qwen4_mtp.md`](qwen4_mtp.md): the opt-in full-resident MTP command. Ordinary
  serving leaves MTP off at every context length.

## Package lifecycle

Model-specific recipe and converted-artifact builders produce a shared
`package_plan` for the writer. Calibration and allocation are explicit inputs.
Serving does not choose a quantization recipe.

- [`artifact_contract.md`](artifact_contract.md): content hashes, versions,
  validation, and fail-closed reads.
- [`artifact_conformance_matrix.md`](artifact_conformance_matrix.md): fields
  carried by each artifact kind.
- [`source_inventory.md`](source_inventory.md): header-only tensor inventory,
  role resolution, and model-family contracts.
- [`probe_evidence.md`](probe_evidence.md): calibration and reconstruction
  evidence.
- [`optimizer_decision.md`](optimizer_decision.md): allocation artifacts and
  their package-plan boundary.
- [`package_format.md`](package_format.md): package plans, manifests, tensor
  formats, shards, sidecars, and integrity declarations.

## Runtime

- [`runtime_resident.md`](runtime_resident.md): manifest-driven loading,
  generation, the OpenAI-compatible HTTP surface, verification, rendering, and
  in-memory prefix reuse.
- [`ssd_streaming.md`](ssd_streaming.md): bounded expert residency, direct
  reads, slot pools, routed decode, and runtime controls.
- [`disk_kv.md`](disk_kv.md): the default-on disk prefix-checkpoint tier
  (restart-warm and cross-session resume), optional DSpark-state companions,
  and the fail-closed restore contract.
- [`tool_calls.md`](tool_calls.md): tool-call dialects, streaming and repair.
- [`diagnostics.md`](diagnostics.md): served, decode and hardware diagnostics,
  interpretation and recording-environment privacy.
- [`completions_api_timing.md`](completions_api_timing.md): client-side timing
  against an existing completions endpoint.

## Correctness and performance work

- [`benchmark_reproduction/`](benchmark_reproduction/): runnable reproduction
  kit for the frozen 48-question generated-answer comparison.
- [`correctness_ladder.md`](correctness_ladder.md): shared package-math checks
  and the role of family-specific served gates.
- [`optimization_methodology.md`](optimization_methodology.md): measurement,
  attribution, A/B discipline, and acceptance rules for runtime work.

Start with the top-level [`README.md`](../README.md) to install and serve.
Contributors should then read [`DEVGUIDE.md`](../DEVGUIDE.md) and
[`AGENTS.md`](../AGENTS.md) before changing the implementation.
