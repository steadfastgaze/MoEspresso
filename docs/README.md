# Documentation

This directory is the public reference for building, validating, and serving
MoEspresso packages. It is self-contained: public users and contributors should
not need private investigation logs to understand the supported paths.

## Model guides

### DeepSeek-V4-Flash

- [`deepseek_v4_package_recipe.md`](deepseek_v4_package_recipe.md): inputs,
  preflight, byte-faithful builds, diagnostics, cache behavior, verification,
  and serving.
- [`deepseek_v4_quality.md`](deepseek_v4_quality.md): Q0/Q1/Q2/Q3 semantics,
  commands, acceptance evidence, and the public/private fixture boundary.
- [`deepseek_v4_speed.md`](deepseek_v4_speed.md): benchmark protocol, current
  resident and streamed records, context envelope, and known gaps.

### Ornith

- [`ornith_package.md`](ornith_package.md): the proven SSD-streaming package
  contract and the missing published-checkpoint source adapter.
- [`ornith_quality.md`](ornith_quality.md): the nine-item served quality gate,
  execution profile, scoring, and private benchmark boundary.
- [`ornith_speed.md`](ornith_speed.md): resident and streamed performance,
  matched reference measurements, context envelope, memory, and caveats.

## Package lifecycle

The two planning routes converge on one `package_plan` before the writer runs:

```text
inventory -> probe -> optimize --\
                                 package_plan -> package -> runtime
GGUF recipe import -------------/
```

- [`artifact_contract.md`](artifact_contract.md): content hashes, versions,
  validation, and fail-closed reads.
- [`artifact_conformance_matrix.md`](artifact_conformance_matrix.md): fields
  carried by each artifact kind.
- [`source_inventory.md`](source_inventory.md): header-only tensor inventory,
  role resolution, and model-family contracts.
- [`probe_evidence.md`](probe_evidence.md): calibration and reconstruction
  evidence.
- [`optimizer_decision.md`](optimizer_decision.md): allocation under size and
  quality constraints.
- [`package_format.md`](package_format.md): package plans, manifests, tensor
  formats, shards, sidecars, and integrity declarations.

## Runtime

- [`runtime_resident.md`](runtime_resident.md): manifest-driven loading,
  generation, the OpenAI-compatible HTTP surface, verification, rendering, and
  in-memory prefix reuse.
- [`ssd_streaming.md`](ssd_streaming.md): bounded expert residency, direct
  reads, slot pools, routed decode, and runtime controls.
- [`disk_kv.md`](disk_kv.md): the default-on disk prefix-checkpoint tier
  (restart-warm and cross-session resume) and its fail-closed restore
  contract.

## Correctness and performance work

- [`benchmark_reproduction.md`](benchmark_reproduction.md): exact artifacts,
  matched controls, timing boundaries, repeat schedule, and quality protocols
  behind the README comparison tables.
- [`correctness_ladder.md`](correctness_ladder.md): shared package-math checks
  and the role of family-specific served gates.
- [`optimization_methodology.md`](optimization_methodology.md): measurement,
  attribution, A/B discipline, and acceptance rules for runtime work.

Start with the top-level [`README.md`](../README.md) to install and serve.
Contributors should then read [`DEVGUIDE.md`](../DEVGUIDE.md) and
[`AGENTS.md`](../AGENTS.md) before changing the implementation.
