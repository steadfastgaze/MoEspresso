# Artifact conformance matrix

What each durable artifact actually carries, mapped against the core contract.
Scope is the five pipeline artifacts (`source_inventory`, `probe_evidence`,
`optimizer_decision`, `package_plan`, `package_manifest`) plus the base contract
every artifact shares.

Two runtime subsystems write durable state outside that pipeline and are not
mapped here. The speculative-decoding release loads DSpark and DFlash drafters
from standalone sidecar folders whose manifests are content-hashed through
`core/artifact.py` as `deepseek_v4_dspark_sidecar` and
`deepseek_v4_dflash_sidecar`. The source tree also defines the
`deepseek_v4_mtp_sidecar` kind as quarantined research code; this release has no
MTP builder command or serving selector. A package that bundles a released
sidecar declares a `drafter` component in its own manifest; the format is in
[`package_format.md`](package_format.md). The disk KV tier keeps checkpoints
and speculative companions under its own versioned on-disk schemas
(`moespresso-disk-kv-v1`, `moespresso-disk-kv-attachment-v1`), documented in
[`disk_kv.md`](disk_kv.md). Neither is a phase of the conversion pipeline, so
neither gets a row.

The cells below that read **n/a** are contracts the pipeline artifacts do not
carry: the manifest declares no KV storage format, no lazy-load grouping, and
no backend-pipeline description, and residency follows from the runtime
adapter kind rather than from a manifest field.

Status words used in the tables:
- **yes**: present and conformant.
- **partial**: present but incomplete, or carried in a different shape than the contract names.
- **no**: required by the contract but not emitted.
- **n/a**: not a contract this artifact carries.

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

## Calibration and allocation artifacts

Model-specific calibrated workflows write `probe_evidence` and
`optimizer_decision` as content-hashed artifacts. Package plans retain their
input identities and per-cell codec decisions.

## package_plan

Built by `package/plan.py` (`make_package_plan`); produced by the
probe/optimizer route (`package_plan_from_decision`), the GGUF recipe builders,
and the converted-artifact IQ_K builder. The manifest builder refuses anything
whose `artifact_kind` is not `package_plan`, so every written package passes
through this artifact.

| Contract field | Status | Note |
|---|---|---|
| producer identity | yes | `producer_kind` (`probe_optimizer`, `gguf_recipe`, `iqk_converted_artifacts`) + `producer_reference` (e.g. the recipe GGUF identity). |
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
| weight formats + transforms + rotation | yes | Per-tensor `format` + `format_params` across the eight on-disk formats (affine: bits/group_size; mxfp4/mxfp8: group 32 + ue8m0 scale identity; kquant: codec + block geometry + module keys for the mlx-kquant installer; iqk: member + wire layout + block geometry + module keys for the mlx-iqk installer; fp16 / f32_passthrough / raw_dtype_passthrough: none). |
| KV formats / lazy-load groups / residency / KV schemas | n/a | The manifest declares none of these. Residency follows from the runtime adapter kind, and the disk KV tier versions its own storage outside the manifest. |
| expert-selection capabilities | partial | `expert_layout` declares stacked/bundled/fused + key suffixes + row order; no selection/offload capability. |
| required backend operations | yes | `required_ops` derived from the tensor formats: `affine_dequant`, `mxfp4_dequant`, `mxfp8_dequant`, `kquant_dequant`, `iqk_dequant`, `fp16_passthrough`, `f32_passthrough`, `raw_dtype_passthrough`. |
| kernel promotion flag | yes | Top-level `optimized_kernels_expected`, copied from the plan; runtime fast paths still validate actual tensor formats and shapes before use. |
| agentic profile identity | partial | Optional `agentic_profile` block (path/sha256/size/family) when the family ships an `agentic_profile.json` sidecar; families without one omit the key. |
| draft-model component identity | partial | Optional `drafter` block when a package bundles a speculative-decoding sidecar: per-file identities, the sidecar's artifact id, and the source package's manifest id. Declared optional under an all-or-nothing contract, so verification covers the drafter bytes when they are present and reports their absence otherwise. |
| load-time validation checks | yes | Manifest fails closed on unwritten tensors / missing shards / an empty plan (`package.empty_plan`) / unsupported or downcast formats; the package verifier checks presence + size + sha256 + declared keys at load. |

## Summary

No open contract violations. The remaining `partial`/`no` rows are either:
- lower-urgency polish (inventory `tensor_id`/`role_owner`/source-file hashes,
  `inputs` chaining through the base field instead of bespoke `source_*_id`,
  `producer.revision`);
- phase-local absences by design (source_inventory and probe_evidence do not own
  token rendering. That identity is established at packaging/runtime); or
- contracts no pipeline artifact carries (KV storage format, lazy-load groups,
  residency, backend pipeline), which the contract says to add if a pipeline
  artifact ever declares them.

Single-family coverage: the core is exercised against MoE (stacked experts,
shared experts, router gate), dense (`ffn.*` roles + a whole-dense-model inventory
fixture with zero experts and zero unknowns), and hybrid/unusual-attention (the
real Qwen layout interleaving linear-attn/SSM and full-attn layers).
