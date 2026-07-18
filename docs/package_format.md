# The mjtq package format

A `mjtq` package is the artifact MoEspresso's converter writes and the runtime
loads: the quantized weights plus an explicit `package_manifest` that tells the
engine *exactly* how to instantiate and run the model. The defining property of
this format is that **the engine never guesses, because the format is fully
explicit**: every fact the runtime would otherwise have to infer (architecture,
on-disk layout, weight transforms, required ops, file integrity, tokenization)
is declared explicitly. Reverse engineering at load time is unnecessary.

This doc documents the subsystem under `src/moespresso/package/` (`plan.py`,
`manifest.py`, `write.py`, `bundle.py`, `hotlist.py`, `agentic_profile.py`, the
shared K-quant modules, and the model builder subpackages `deepseek_v4/` and
`qwen/`) plus the resulting on-disk layout.

## What "mjtq" means: MoEspresso Jang TurboQuant

`mjtq` = **MoEspresso Jang TurboQuant**. It *reuses* jang's TurboQuant codec and tensor
conventions for compression (the `.tq_packed` / `.tq_norms` / `.tq_bits`
arrays, Hadamard rotation with a seed, per-row norms, bit-packing), but it adds
a strict layer on top: the explicit `package_manifest` is the contract the
runtime reads.

This is **distinct from the third-party `jangtq` format**.
A `jangtq`-shaped package is consumed by *parsing* `jang_config.json` and then
*knowing* a pile of conventions: expert shard
naming, the TQ packing layout, which tensors are fp16 passthrough, the double
`model.language_model.` key nesting. Every one of those "knows" is a place a new
model can make the engine guess wrong: exactly the failure mode this project
exists to kill. In mjtq, jang supplies the compression backend. The manifest
defines the package format and replaces inference with declaration.

The format identity lives in `manifest.py`:

- `PACKAGE_FORMAT = "mjtq"`, `PACKAGE_FORMAT_VERSION = 1`
- `PACKAGE_FORMAT_FEATURES = frozenset({"calibration"})`. mjtq declares its
  strictness here. The generic convert orchestrator reads that declaration. A mjtq
  package's probe evidence must be activation-weighted by a real imatrix; an
  uncalibrated mjtq is a red flag the spec names. Other package formats declare
  their own feature sets, and the convert pipeline consults only those
  declarations when deciding whether calibration is required.

## The `package_plan`: the writer-facing allocation

Two build routes produce packages: the probe/optimizer route
(`moespresso-convert`) and the GGUF K-quant recipe route
(`moespresso-ds4-kquant-package`, `moespresso-qwen-kquant-package`, with the
per-model recipe mapping in `package/deepseek_v4/recipe.py` and
`package/qwen/recipe.py` and the shared GGUF parsing in `kquant_recipe.py`).
Both converge on one artifact before anything is written: the `package_plan`
(`plan.py`). The plan carries the normalized per-tensor allocation, the
producer identity (`producer_kind`, `producer_reference`), the chained
`source_decision_id`/`source_probe_id` (null on the recipe route), the
`optimized_kernels_expected` promotion flag, and any explicit force overrides
(`--force-format PATTERN=FORMAT`), which fail closed on unknown formats and
unmatched patterns unless explicitly allowed and support a dry-run preview.
The writer and the manifest builder consume only the plan; neither branches on
which route produced it.

## The `package_manifest`: the package's full self-description

`build_package_manifest(...)` in `manifest.py` is **pure**: a function of the
`package_plan` + the source architecture config + the list of written-file
identities. It refuses any input whose `artifact_kind` is not `package_plan`.
No mlx, no jang, no weight bytes; fully testable without a compute backend.
The heavy packing and safetensors writing live in `write.py` (the imperative
shell, using the standard runtime dependencies). The manifest is itself a content-addressed
artifact, so the runtime can verify `manifest.artifact_id` before trusting it.

The manifest declares, explicitly:

### Copied architecture facts

`_architecture()` carries the **complete** text config the runtime builds the
graph from, so the engine instantiates the model from the manifest alone and
never reads the source `config.json` (the spec's "runtime never performs source
archaeology"). A trimmed field set was the original bug: it discarded the
linear-attn / SSM fields the graph needs and forced the loader back to
`config.json`.

- `family` selects the model class; `config` is everything that class's
  `ModelArgs.from_dict` needs. The runtime also derives the served context
  limit from this embedded config (`max_position_embeddings`, already scaled
  for position-scaled families), so the manifest needs no bespoke
  context-limit field.
- A readable `_ARCH_SUMMARY_FIELDS` subset (`num_hidden_layers`, `hidden_size`,
  `num_experts`, `num_experts_per_tok`, `layer_types`, `moe_intermediate_size`,
  …) is duplicated at top level so a glance shows the shape, but the runtime
  still builds from the full `config`.
- `modality: "text"` plus a declared `excludes` list: mjtq serves the text
  model only, and **says so**. The source may be a VL/MTP checkpoint; the
  support scope is a declared fact, never a silent assumption. Qwen-family
  packages exclude `["vision", "mtp"]`; DeepSeek-V4 Flash excludes `["mtp"]`.
  A future vision mjtq would declare a different modality and carry the vision
  config.
- `source_nesting` declares the key nesting instead of baking it into the
  loader: `"model.language_model."` for the Qwen families, `""` for
  DeepSeek-V4 Flash. DS4 packages additionally carry the family profile's
  structural facts (layer kinds, compression ratios, per-layer rope/YaRN,
  router, cache policy, prompt renderer).
- Smoke artifacts: `max_experts` clamps `config.num_experts` (and
  `num_experts_per_tok`) so the served graph has exactly the experts on disk; a
  reduced-expert smoke is a *declared*, smaller model for crash/coherence
  checks, recorded as `smoke_max_experts`.

### Per-tensor on-disk format + `format_params`

`tensors` carries one entry per packed tensor (`_tensor_entry` /
`_passthrough_entry`): `source_name`, `role` (the typed vocabulary), `kind`
(`expert | affine | fp16_passthrough | raw_dtype_passthrough | passthrough`),
the on-disk location (`shard` file + `key_prefix`), and the weight format with
its params. Eight formats exist:

- **`tq`** (routed experts): `format_params = {tq_version, bits, seed}`. The TQ
  transform is declared by **versioned reference**: the engine knows what
  `tq_version 1` means (Hadamard rotation with seed, per-row norms, bit-packing),
  but versioned so it cannot silently drift. `format_params` is a sub-object
  precisely so the transform can later be declared *structurally*
  (`hadamard_rotate(seed)`, `pack_bits(bits)`, `scale(norms)`) without a major
  version bump: declare enough to be unambiguous and verifiable now, leave room
  to grow.
- **`affine`** (dense): `format_params = {bits, group_size}`.
- **`mxfp4`** (dense or routed experts): fixed group size 32 with uint8 UE8M0
  scales; `format_params` records `source_codec` (e.g. `fp4_e2m1_ue8m0` for
  DS4 source-FP4 experts carried losslessly) and a `lossless` flag.
- **`mxfp8`** (dense): same group-32/UE8M0 identity at 8 bits.
- **`kquant`** (dense or routed experts): a GGUF K-quant codec by name.
  `format_params` records the codec plus its block geometry (`bits`,
  `group_size`, `bytes_per_block`, `weights_per_block`, from
  `kquant_format.KQUANT_GEOMETRY`) and the `imatrix_key` for imatrix-steered
  codecs; the entry also carries the `module_weight_key`/`module_path` the
  mlx-kquant installer needs. The manifest builder fails closed on a codec
  outside the registry or a missing module key.
- **`fp16`** (passthrough): `format_params = {}`. The array is stored as
  float16.
- **`f32_passthrough`**: the array is stored as float32, verbatim.
- **`raw_dtype_passthrough`**: the array is stored in its source dtype,
  verbatim. Control-tensor roles (attention sinks, positional-encoding
  companions, router bias and id maps, hyper-connection controls) are forced
  to this format; a manifest that declares any of them at a downcast format is
  refused with a blocking `package.control_tensor_downcast` validation.

Routed experts accept `tq`, `mxfp4`, or `kquant`; dense tensors accept
`affine`, `mxfp4`, `mxfp8`, or `kquant`; anything else is a blocking
validation.

`passthrough` tensors (structural norms, SSM state, conv1d) are stored
verbatim in source (pre-sanitize) form. This is load-bearing: e.g. `conv1d.weight`
must stay `[out, 1, k]` so mlx_lm's qwen3_5 sanitize fires its coupled
transpose + RMSNorm `+1.0` shift at load (see `_passthrough_array` in
`write.py`, which handles the fp16, f32, and raw-dtype forms). They flow from
the inventory. Keeping them out of the plan preserves optimizer purity.

### File identities (path + size + sha256): fail closed

`file_identity()` records `{path, size_bytes, sha256}` for every written shard
and copied package member. The on-demand `moespresso-verify` gate re-hashes
every declared shard, tokenizer file, and agentic profile and fails if a file
is missing, the size differs, or the sha256 differs. Run that gate after every
build, download, copy, or move and before loading an unverified package. The
serve path does not repeat a tens-of-gigabytes hash pass at every startup.

### Required backend ops

`required_ops` is the sorted set the engine must support, derived from the
tensor formats actually present: `tq → tq_dequant`, `affine → affine_dequant`,
`mxfp4 → mxfp4_dequant`, `mxfp8 → mxfp8_dequant`, `kquant → kquant_dequant`,
`fp16 → fp16_passthrough`, `f32_passthrough → f32_passthrough`,
`raw_dtype_passthrough → raw_dtype_passthrough`. The runtime adapter selection
(`runtime/build.py`) keys off `required_ops` + `family`: `deepseek_v4_flash`
builds the `mjtq_dsv4` adapter, a dense `qwen3_5_dense` whose ops stay within
the dense affine set builds `regular_jang_v2`, a `qwen3_5_moe` package with
K-quant experts builds `qwen_kquant_moe`, and any other family with
`tq_dequant` builds `jangtq_moe`. An unrecognized combination raises
`UnsupportedRuntimeAdapter` rather than guessing.

The manifest also carries a top-level `optimized_kernels_expected` flag
(default false), copied from the plan. Setting it is an explicit
package-build promotion; runtime fast paths still validate actual tensor
formats and shapes before use.

### Manifest and on-demand validation checks

The manifest carries `status` (`valid` / `invalid`) and a list of `Validation`
entries; any `blocking` entry means the package must not load. The builder emits
blocking validations for, among others:

- `package.unwritten_tensor`: an allocation references a tensor with no written
  location.
- `package.missing_shard`: a tensor's shard isn't in the written-files set.
- `package.empty_plan`: the plan has no allocation (infeasible) so there is
  nothing to package.
- `package.unsupported_expert_format` / `package.unsupported_dense_format` /
  `package.unsupported_kquant_codec` / `package.control_tensor_downcast`: a
  format outside the declared vocabulary for its tensor class.

`moespresso-verify` adds the integrity layer: it validates the manifest's
content id, status, package-format version, and embedded blocking findings;
checks declared package-member identities; and confirms that every tensor's
expanded keys (`expected_keys()`, expanding `key_prefix` by format) exist in a
manifest-declared safetensors shard. It also rebuilds `config.json` and
`jang_config.json` from the manifest and compares their semantics. These checks
run on demand and stay outside the normal model-load path.

### Tokenizer / rendering identity

The `tokenizer` block (built in `package/tokenizer.py`) records the copied
tokenizer files (path/size/sha256), `has_tokenizer`, `chat_template_source`, and
a **`rendering_id`**: a sha256 over the tokenizer file identities, computed
*after* any MoEspresso-owned chat template is installed, so the hash covers the
exact template that ships. `rendering_id` is a runtime cache contract: the
prefix cache keys on it (`runtime/prefix_cache.py`,
`runtime/http.py:rendering_identity`) so a byte-prefix never drifts across a
template change. The runtime tokenizes from the package, never the source.

`provenance` chains the package back to its inputs: `source_plan_id`
(== `plan.artifact_id`), plus the `source_decision_id` and `source_probe_id`
the plan copied through, and a `provenance.package_plan` block recording the
producer identity, the promotion flag, and any force overrides.

### The agentic profile sidecar

Families with recorded agent-loop evidence ship an `agentic_profile.json`
sidecar (`package/agentic_profile.py`), written beside the vendored chat
template and registered in the manifest as an `agentic_profile` identity block
(path, sha256, size, family). The profile records how an agent loop should
drive the model: the tool-call dialect it emits reliably, whether the repair
layer is load-bearing for that dialect, the thinking flag for tool work, the
re-prompt policy, and recommended sampling defaults. `agentlib` reads the file
and configures its loop from it; a family without recorded evidence gets no
file, and a missing file means the client decides everything. Readers fail
closed on a schema version above the one they support.

#### Agentic profile records

The shipped family profiles use promoted results from served studies:

- Ornith 1.0 35B uses the DSML dialect with repair required. The recorded study
  completed all 15 tasks with repair enabled. The observed quoting malformations
  were salvaged, and any unsalvaged repair remains an alarm condition. The profile
  records the sampling settings exercised by that study.
- DeepSeek-V4-Flash uses the DSML dialect with repair optional. Its road-test
  campaign produced 40 tool requests with no malformations. The campaign did not
  establish sampling defaults, so the profile leaves sampling to the client.

## On-disk layout

A mjtq package directory contains:

- `model-NNNNN-of-COUNT.safetensors` shard(s). `write.py` streams within every
  tensor (a row-band for affine/fp16, one expert at a time for TQ) so a 35B
  model converts in bounded RAM, and starts a new shard once a byte cap
  (`--shard-size-gb`) is passed. The final count is unknown until the end, so
  shards are written `-of-?????` and renamed `-of-COUNT` once done. A "tensor
  group" (all keys for one source tensor) is added atomically, so a group never
  straddles two shards and a per-tensor read stays within one file. Each shard's
  `__metadata__` carries `{"format": "mjtq"}` (plus expert-bundle geometry,
  below). Shard bytes are **deterministic**: `_write_shard_deterministic`
  serializes the header with sorted keys and fixed alignment padding, keeping
  the library's data layout, so identical inputs always produce identical
  shard files and hashes. The library serializer keeps `__metadata__` in
  per-instance hash-map order, which would make two builds of identical
  content hash differently.
- The mjtq `package_manifest` artifact (the contract above).
- `expert_hotlist.json` (cold-start hotlist, below) when the source is a routed
  MoE with imatrix counts.
- `agentic_profile.json` (above) for families with a profile of record.
- Generated jang-compatible sidecars (`config.json`, `jang_config.json`;
  written by `sidecars.py`): a compat view for the loader, generated from the
  manifest, with the manifest staying the source of truth.
- Copied aux files: tokenizer, chat template, `preprocessor_config.json`, etc.

Key conventions per format:

- routed experts → one per-layer bundle `...switch_mlp.experts.tq_bundle`
- affine → `<base>.weight` / `.scales` / `.biases`
- fp16 passthrough → the raw array under its own name

## The per-expert bundle layout (routed experts)

`bundle.py` is the single source of truth for the **streaming bundle format** and
the reason a streamed expert miss is cheap. Instead of six stacked per-projection
tensors per routed layer, mjtq writes **one uint8 bundle tensor per layer**:

```
...switch_mlp.experts.tq_bundle   →   uint8 [n_experts, row_bytes]
```

Row `e` concatenates expert `e`'s **full payload, contiguous**, one component
pair per projection in a fixed per-codec order (`row_order_for_codecs`). The
components depend on each projection's declared codec:

```
tq      -> [ packed | norms ]     (packed uint32, norms float16)
mxfp4   -> [ packed | scales ]    (packed uint32, scales uint8 UE8M0)
kquant  -> [ weight | scales ]    (both uint8 wire bytes)
```

so an all-TQ layer's row reads
`[ gate.packed | gate.norms | up.packed | up.norms | down.packed | down.norms ]`,
and a mixed-codec layer substitutes each projection's own pair. The projection
codec is declared in the bundle metadata; readers never infer it from bit
width. A missed expert costs **one contiguous pread** because the bundle layout
removes the six-way seek scatter. The row stride is the exact
component sum with **no padding**: plain pread needs no alignment, and direct
IO is explicitly out of scope.

The geometry contract (per component: within-row `offset`, `nbytes`, per-expert
`shape`, `dtype`, plus the per-projection codec and `bits`) travels in the
**shard's safetensors `__metadata__`** under `expert_bundles` as versioned
JSON. This keeps the expert index header-only: no weight reads, no separate
manifest file to locate an expert. `assemble_layer_bundle()` writes it,
`decode_bundle_metadata()` reads it, and validation is **exact-tiling**: every
component range must follow the declared row order back-to-back and the last
must end exactly at `row_bytes`. Because the format has no padding, any gap or
overlap means writer/reader drift and fails loud, never guessed around.
`component_array()` is the reader-side slice for the correctness ladder and
inspector probe.

The manifest's `expert_layout` block (`_DEFAULT_EXPERT_LAYOUT`) names this
convention: `bundled: True`, `fused_gate_up: True` (the source `gate_up_proj`
splits into gate + up sub-projections), `key_suffixes: ["tq_bundle"]`, and the
same `row_order`. Older stacked packages (`tq_packed` / `tq_norms` / `tq_bits`)
are **not readable**: the runtime fails loud with a re-convert message rather
than guess at an old layout.

## The cold-start expert hotlist

`hotlist.py` bakes a cold-start expert hotlist into the package from the
calibration imatrix's per-layer routed-expert usage counters. The calibration
imatrix is already a mandatory, provenance-recorded convert input, and it
carries ~millions of routed calibration tokens per layer.

`build_package_expert_hotlist()` ranks each layer's experts by count and emits
`expert_hotlist.json`. Measured against real request demand: seeding capacity-70
from these counts captures a median 0.40 of a request's expert-demand mass vs
0.27 for arbitrary seeding, with zero run history. This artifact is the **floor
for the first request on a host with no saved demand history**. A runtime-saved
demand hotlist captures ~0.60 and takes precedence when present. The emitted file uses the
**same schema** as the saved-demand hotlists, so the streaming builder
(`ssd_streaming_build.load_expert_hotlist`) consumes either interchangeably (it
caps installed priors so neither can dominate live traffic).

**Fail-closed alignment.** imatrix counts are keyed by GGUF block index; the
package's routed layers are keyed by model layer index. These have coincided on
every artifact checked, but a mismatch would silently seed the *wrong* layers'
experts. So the builder requires **exact layer-set equality** with the package's
expert index (and at least `num_experts` counters per layer) and raises
`HotlistAlignmentError` (emitting nothing, with a loud reason) otherwise.
`write_package_expert_hotlist()` returns 0 (writes nothing) for a dense model or
a package with no routed experts; the convert caller logs an alignment failure
and proceeds without a hotlist rather than shipping a wrong one.

## Why this matters

Every fact above is in the manifest or shard metadata and is declared at
package time. The on-demand verifier confirms content identity and generated
sidecar semantics; the runtime then instantiates the architecture from the
manifest's `config`, maps keys to modules via `expert_layout`, dequantizes via
the declared `format` + `format_params`, checks `required_ops`, and tokenizes
via the declared `rendering_id`. **The engine never guesses, because the format
is fully explicit.**
