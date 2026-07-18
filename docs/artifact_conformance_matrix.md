# Artifact conformance matrix

What each durable artifact actually carries, mapped against the core contract.
Scope is the five pipeline artifacts (`source_inventory`, `probe_evidence`,
`optimizer_decision`, `package_plan`, `package_manifest`) plus the base contract
every artifact shares. Speculation, KV-state, backend-pipeline, and residency
contracts are out of scope until those phases exist. Their absence here is
expected.

Status words used in the tables:
- **yes**: present and conformant.
- **partial**: present but incomplete, or carried in a different shape than the contract names.
- **no**: required by the contract but not emitted.
- **n/a**: not yet in scope (future phase).

## Base contract (every artifact)

Defined in `core/artifact.py` (`make_artifact`, `validate_base`, `canonical_json`).

| Required key | Status | Note |
|---|---|---|
| `artifact_kind` | yes | Set by `make_artifact`; fail-closed on unknown kind (`ArtifactError`). |
| `schema_version {major,minor}` | yes | Major mismatch fails closed in `validate_base`. |
| `artifact_id` (content hash) | yes | sha256 of canonical JSON, kind-prefixed (`inv:`/`probe:`/`dec:`/`plan:`/`pkg:`), self-excluded from the hash. |
| `producer {tool,version,...}` | partial | `tool`+`version` present per phase; no `revision`/`command`. |
| `created_at` (UTC) | yes | Stamped at `write_artifact` (caller supplies the string; no wall-clock read) and excluded from the content hash via `_HASH_EXCLUDED`, so persisting never perturbs the id. |
| `inputs` (consumed artifact ids/hashes) | partial | Field exists, defaults to `[]`. The phases chain provenance through dedicated fields (`source_inventory_id`, `source_probe_id`, `source_decision_id`, `provenance.source_plan_id`) instead of populating `inputs`. Provenance is chained, just not through this list. |
| `subject` | yes | Threaded through every phase (inventory builds it; later phases reuse it). |
| `required_features` | yes | Base field, empty default. `validate_base` fails closed on a feature absent from `KNOWN_FEATURES`. Calibrated probe evidence declares `calibration` through it. |
| `optional_annotations` | no | Not emitted. Lower priority (fail-open, ignorable). |
| `status` | yes | One of draft/valid/invalid/superseded/retired; a bad status fails closed. |
| `validation` (structured entries) | yes | `Validation` dataclass: severity/code/message/path/phase/blocking, optional expected/actual. |

Canonicalization rules: sorted-key UTF-8 JSON, NaN/Inf forbidden (`_assert_finite`),
integers for shapes/bytes, file identity `{path,size_bytes,sha256}`, all **yes**.
The contract's tensor-list sort key `(layer_index, expert_index, role, source_name)`
is **partial**: artifacts sort by `(layer_index, projection, source_name)` with no
`expert_index` (experts are stacked and addressed by projection), close but not
identical to the contract key.

## source_inventory

Built by `inventory/build.py` (`build_inventory_from_headers`).

| Contract field | Status | Note |
|---|---|---|
| `source` (files, sizes, hashes, format, config) | partial | `subject` carries `source_root` + `source_format`; per-file size/sha256 are not recorded here (they live in the package manifest). |
| `tokenizer_rendering_id` | no | This phase does not produce it. Packaging/runtime establishes active tokenizer and rendering identity. |
| `architecture_candidates` | no | Candidate detection and confidence are absent; the family is implicit. |
| `tensors[]` records | partial | Each entry carries `source_name`, `role`, `kind`, `layer_index`, `shape`, `dtype`, `shard`, `gguf_keys`, `status` (plus `projection` for experts). Missing: `tensor_id`, `role_owner`, `expert_index`, byte-range/hash. |
| `expected_tensors` / `role_map` | partial | Classification covers required/affine/expert/passthrough/unknown via `counts`; no explicit generated/passthrough/unexpected taxonomy. |
| `validation` | yes | Emits unknown-tensor warnings and per-key imatrix-coverage warnings (`imatrix.key_absent`, `inventory.unknown_tensors`), with an `imatrix_coverage` summary. |

## probe_evidence

Built by `probe/build.py` (`build_probe_evidence`).

| Contract field | Status | Note |
|---|---|---|
| source inventory id | yes | `source_inventory_id`, taken from the inventory artifact. |
| tokenizer/rendering identity | no | Not produced by this phase; token rendering is package/runtime state. |
| calibration dataset identity (name, source, size, hash, sampling) | yes | `calibration` block from the calibration provider (its identity dict); calibrated evidence declares `required_features=["calibration"]`. The uniform path is an explicit opt-in stamped `calibration={"kind":"uniform"}` and declares no required feature, so it can never pass as calibrated. |
| seeds | yes | `config.seed`. |
| sample counts | yes | `config.expert_sample`, `config.sample_rows`; per-expert `sampled` count on each expert unit. |
| metric definitions | partial | Per-unit quality tables are present; code documents the activation-weighted reconstruction metric, while the artifact omits its definition. |
| variance / confidence | no | Not recorded. |
| regression thresholds | no | Not recorded. |
| hardware / runtime context | no | Not recorded (only a `config.fp16_measurement` flag). |
| tensor groups measured + metrics | yes | `units[]`: per-tensor quality-vs-bits tables, plus importance and `imatrix_mapped`. |
| candidate quant / storage settings | yes | `config.expert_bits`, `config.affine_bits`, `config.affine_group_sizes`. |
| failures / skips | partial | Coverage warnings (`probe.no_imatrix` when a kind mapped nothing, `probe.partial_imatrix` when it mapped some) and a `coverage` summary; no per-tensor skip list. |

## optimizer_decision

Built by `optimize/decide.py` (`decide`).

| Contract field | Status | Note |
|---|---|---|
| constraints | yes | `constraints`: target_quality / target_size_gb / tau / alpha / lm_head_bits (plus any role weights, budget split, expert-importance normalization). |
| objective | yes | `objective` string, e.g. "maximize importance-weighted fidelity F per byte s.t. ...". Names the actual goal and constraints. |
| selected weight formats + transforms | yes | `allocation[]`: per-source tq / affine / fp16_passthrough with bits/group_size. |
| selected KV formats | n/a | No KV phase yet. |
| expert retention / residency | partial | Per-layer x projection bits in `allocation`; no residency/retention policy yet. |
| expected memory | yes | `achieved.size_gb`, `achieved.expert_size_gb`, `achieved.tensor_size_gb`, `achieved.bps`. |
| expected quality / risk | yes | `achieved.fidelity`, `achieved.worst_layer_tail`. |
| fallback choices | no | Not recorded. |
| rejected alternatives + reasons | yes | `rejected[]` of `{choice, reason}`: records genuine rejections only (e.g. quantizing fp16-passthrough gates). No fabricated search the greedy did not run. |
| provenance (optimizer vs manual) | partial | `producer` identifies the tool; no explicit `provenance.mode`. |
| inputs chaining | partial | `source_probe_id` chains the probe; not via base `inputs`. |

An infeasible run is recorded as a valid artifact describing infeasibility
(`feasibility` set, empty `allocation`, `achieved` null, blocking validation entry,
`status="invalid"`).

## package_plan

Built by `package/plan.py` (`make_package_plan`); produced by both the
probe/optimizer route (`package_plan_from_decision`) and the GGUF recipe builders.
The manifest builder refuses anything whose `artifact_kind` is not
`package_plan`, so every written package passes through this artifact.

| Contract field | Status | Note |
|---|---|---|
| producer identity | yes | `producer_kind` (optimizer vs recipe) + `producer_reference` (e.g. the recipe GGUF identity). |
| allocation | yes | `allocation[]`: the normalized per-tensor rows the writer consumes. |
| force overrides | yes | `force_overrides[]` (`pattern`/`target` pairs); overrides fail closed on unknown formats and unmatched patterns unless explicitly allowed, and support a dry-run preview. |
| kernel promotion flag | yes | `optimized_kernels_expected` (default false); copied into the manifest. |
| inputs chaining | yes | `source_decision_id` + `source_probe_id` copied through from the producing route (null for pure recipe imports). |
| constraints / achieved | yes | `source_constraints` and `achieved` carried for provenance. |

## package_manifest

Built by `package/manifest.py` (`build_package_manifest`); tokenizer block by
`package/tokenizer.py` (`copy_tokenizer_into_package`).

| Contract field | Status | Note |
|---|---|---|
| package format version | yes | `package_format` ("mjtq") + `package_format_version` (integer) are emitted as explicit fields. |
| architecture | yes | `architecture` copies the complete text config (so the runtime builds the graph from the manifest alone) plus a readable summary, `family`, `modality`, and declared `excludes`. |
| source inventory id | partial | Chained via probe -> decision -> plan (`provenance.source_probe_id`); not a direct manifest field. |
| plan / decision id | yes | `provenance.source_plan_id` is the primary key; `source_decision_id` and `source_probe_id` are copied through the plan (null on the recipe route). A nested `provenance.package_plan` block records `producer_kind`, `producer_reference`, `optimized_kernels_expected`, and `force_overrides`. |
| tokenizer / rendering identity | yes | `tokenizer` block: installed tokenizer file identities + `rendering_id` (sha256 over the installed tokenizer files, including the chat template) + `chat_template_source`. Runtime cache keys additionally fold in resolved chat-template kwargs via `runtime.http.rendering_identity(...)`. |
| tensor files + layouts | yes | `files[]` with path/size/sha256; each `tensors[]` entry carries `shard` + `key_prefix`. |
| weight formats + transforms + rotation | yes | Per-tensor `format` + `format_params` across the eight on-disk formats (tq: tq_version/bits/seed; affine: bits/group_size; mxfp4/mxfp8: group 32 + ue8m0 scale identity; kquant: codec + block geometry + module keys for the mlx-kquant installer; fp16 / f32_passthrough / raw_dtype_passthrough: none). |
| KV formats / lazy-load groups / residency / KV schemas | n/a | Future phases. |
| expert-selection capabilities | partial | `expert_layout` declares stacked/bundled/fused + key suffixes + row order; no selection/offload capability. |
| required backend operations | yes | `required_ops` derived from the tensor formats: `tq_dequant`, `affine_dequant`, `mxfp4_dequant`, `mxfp8_dequant`, `kquant_dequant`, `fp16_passthrough`, `f32_passthrough`, `raw_dtype_passthrough`. |
| kernel promotion flag | yes | Top-level `optimized_kernels_expected`, copied from the plan; runtime fast paths still validate actual tensor formats and shapes before use. |
| agentic profile identity | partial | Optional `agentic_profile` block (path/sha256/size/family) when the family ships an `agentic_profile.json` sidecar; families without one omit the key. |
| load-time validation checks | yes | Manifest fails closed on unwritten tensors / missing shards / an empty plan (`package.empty_plan`) / unsupported or downcast formats; the package verifier checks presence + size + sha256 + declared keys at load. |

## Summary

No open contract violations. The remaining `partial`/`no` rows are either:
- lower-urgency polish (inventory `tensor_id`/`role_owner`/source-file hashes,
  `inputs` chaining through the base field instead of bespoke `source_*_id`,
  `producer.revision`);
- phase-local absences by design (source_inventory and probe_evidence do not own
  token rendering. That identity is established at packaging/runtime); or
- correctly-deferred future phases (KV-state, residency, speculation,
  backend-pipeline), which the contract says to add when they become real.

Single-family coverage: the core is exercised against MoE (stacked experts,
shared experts, router gate), dense (`ffn.*` roles + a whole-dense-model inventory
fixture with zero experts and zero unknowns), and hybrid/unusual-attention (the
real Qwen layout interleaving linear-attn/SSM and full-attn layers).
