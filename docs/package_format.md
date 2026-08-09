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

Three build routes produce packages: the probe/optimizer route
(`moespresso-convert`), the GGUF K-quant recipe route
(`moespresso-ds4-kquant-package`, `moespresso-qwen-kquant-package`, with the
per-model recipe mapping in `package/deepseek_v4/recipe.py` and
`package/qwen/recipe.py` and the shared GGUF parsing in `kquant_recipe.py`),
and the converted-artifact route (`moespresso-ds4-iqk-package`), which packages
routed-expert bytes a separate conversion stage already encoded.
All converge on one artifact before anything is written: the `package_plan`
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
its params. Nine formats exist:

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
- **`iqk`** (routed experts or dense): an IQ_K block codec by member name.
  `format_params = {iqk_codec, layout, bits, ggml_type, weights_per_block,
  bytes_per_block, row_meta_bytes}` plus the `module_weight_key`/`module_path`
  the installer needs. Every scale lives inside the row, so a projection stores
  one component, `blocks` (uint8, `[out_features, bytes_per_row]` per expert),
  where `bytes_per_row = row_meta_bytes + (in_features / 256) * bytes_per_block`
  from `iqk_format.IQK_GEOMETRY`. The rate therefore depends on the row width
  and no member has a single bits-per-weight. The member is a per-(layer,
  role) fact: a package built from a mixed allocation carries different members
  in one layer and nothing downstream reads a package-wide bit width. `layout`
  is `ik_wire` for the quantizer's own row-major byte stream, or `iqk_relayout`
  once a build step has rearranged the same encoded bytes for the decode
  kernels; a reader that understands one refuses the other. The manifest
  builder fails closed on a member outside the registry and on an unknown
  layout. The two layouts spend the same bits, so a relayout row is exactly as
  wide as the wire row it replaces and both live inside the same `blocks`
  component with no shape, offset, or row stride moving; a relayout row holds
  its own streams end to end in the order `mlx_iqk.format` declares, so
  element `[r]` of `blocks` is still row `r`'s payload.
  `moespresso-ds4-iqk-relayout` moves a built package between them without
  re-encoding, writes new plan and manifest ids, and gates the move by turning
  every rewritten row back into wire bytes and by decoding sampled rows through
  both references. Only `iqk_relayout` serves. The target expert pool is
  described in `docs/ssd_streaming.md`; resident sidecars and dense IQ_K are
  described in `docs/runtime_resident.md`.
  A dense tensor may also declare `iqk`, restricted to the 4- to 6-bit members
  (`iqk_format.IQK_DENSE_MEMBERS`: `iq4_ks`, `iq4_k`, `iq5_k`, `iq6_k`); a
  routed-only member on a dense tensor is a blocking validation. No package
  declares dense IQ_K today and the dense relayout step does not exist, so the
  dense serving routes described in `docs/runtime_resident.md` guard an
  unproven path.
- **`fp16`** (passthrough): `format_params = {}`. The array is stored as
  float16.
- **`f32_passthrough`**: the array is stored as float32, verbatim.
- **`raw_dtype_passthrough`**: the array is stored in its source dtype,
  verbatim. Control-tensor roles (attention sinks, positional-encoding
  companions, router bias and id maps, hyper-connection controls) are forced
  to this format; a manifest that declares any of them at a downcast format is
  refused with a blocking `package.control_tensor_downcast` validation.

Routed experts accept `tq`, `mxfp4`, `kquant`, or `iqk`; dense tensors
accept `affine`, `mxfp4`, `mxfp8`, `kquant`, or `iqk` at a dense member;
anything else is a blocking validation.

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
`iqk → iqk_dequant`, `fp16 → fp16_passthrough`,
`f32_passthrough → f32_passthrough`,
`raw_dtype_passthrough → raw_dtype_passthrough`. The runtime adapter selection
(`runtime/build.py`) keys off `required_ops` + `family` and resolves to one of
four kinds: `deepseek_v4_flash` builds the `mjtq_dsv4` adapter; a dense
`qwen3_5_dense` whose ops stay within the dense affine set builds
`regular_jang_v2`; a `qwen3_5_moe` package carrying `kquant_dequant` without
`tq_dequant` builds `qwen_kquant_moe`; and any other family with
`tq_dequant` builds `jangtq_moe`. An unrecognized combination raises
`UnsupportedRuntimeAdapter` rather than guessing. The kind also selects the
top-level model builder. `jangtq_moe` and `qwen_kquant_moe` take the shared
pooled builder. The `mjtq_dsv4` builder installs routed IQ_K experts into the
same persistent pool internally: capacity equal to the expert count is fully
resident, while smaller capacities stream missing bundle rows. Selected
layers can grow through the runtime's post-request pool transaction.
The dense adapter uses the resident builder in `runtime/serve.py`. Runtime
details are in `docs/runtime_resident.md` and `docs/ssd_streaming.md`.

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
  `package.unsupported_kquant_codec` / `package.unsupported_iqk_codec` /
  `package.unsupported_iqk_layout` / `package.control_tensor_downcast`: a
  format, codec, or wire layout outside the declared vocabulary for its tensor
  class.

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

- Ornith 1.0 35B uses the native dialect, the XML its vendored chat template
  teaches, with repair optional. A structural emission battery of eleven
  write-shaped requests carrying 4-45 KB parameter values was scored on the
  shipping K-quant package and an unquantized Q8_0 reference at a greedy and a
  sampled profile. Every native arm came back with zero structural defects and
  zero strict-parse rejections, at 8 to 11 usable calls out of 11 per pass, and
  the misses are elicitation rather than emission. Teaching DSML in-context
  instead drops the closing quote after the parameter name, so the name
  attribute swallows the following `string=` and the parser sees one attribute
  where two were taught. That hits 73 to 81 per cent of parameter opens on both
  artifacts, at emission offsets 28 to 1607, which is the parameter's opening
  tag rather than deep in a long value, and no DSML arm produced a single usable
  call through the full parse and repair path. The unquantized reference is the
  worse of the two, which rules the quantization level out and leaves the
  dialect as the cause.
  Repair stays optional because it has nothing to fix at native and cannot
  rescue DSML. The thinking flag, the re-prompt policy and the sampling
  settings carry over from the earlier loop study that recorded them; the
  battery ran a greedy and a higher-temperature profile and the dialect result
  is the same at both.
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
- The bundled drafter's shards and sidecar manifest, when the package declares
  a `drafter` component (below).
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
tq      -> [ packed | norms ]        (packed uint32, norms float16)
mxfp4   -> [ packed | scales ]       (packed uint32, scales uint8 UE8M0)
kquant  -> [ weight | scales ]       (both uint8 wire bytes)
iqk     -> [ blocks ]                (uint8 wire bytes; scales live in the row)
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

IQ_K target experts use this bundle directly through the pooled runtime. A
miss reads one `blocks` row and splits its declared relayout streams into the
selected slot. At capacity equal to the declared expert count, the default
startup policy loads every row and runs the zero-miss identity-slot path through
the same `PooledSwitchGLU` graph. Smaller capacities retain the same graph and
replace slot contents from bundle rows on demand.

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

## Draft-model sidecar manifests

Speculative-decoding drafters load from standalone sidecar folders (builders
in `package/deepseek_v4/dspark_sidecar.py`, `mtp_sidecar.py`, and
`dflash_sidecar.py`; usage in `docs/speculative_decoding.md`). A sidecar is
not a mjtq package. It carries its own manifest (`dspark_sidecar.json` /
`mtp_sidecar.json` / `dflash_sidecar.json`), content-hashed through the
`core/artifact.py` helpers, with a per-tensor format table and per-file
sha256 provenance. The loaders (`runtime/deepseek_v4/dspark_load.py`,
`mtp_load.py`, `dflash_load.py`) validate the manifest fail-closed, quantize
the module tree per the manifest, strip the constructed skeleton
(`spec_decode.strip_draft_skeleton` swaps every lazy random-init parameter
for a zero-stride placeholder so no evaluation between construction and the
strict load can allocate the full float32 tree), and strict-load the
weights.

The DSpark builder resolves the drafter stage count from a declaration
only: top-level `config.json` `n_mtp_layers` first, the checkpoint's
`inference/config.json` second, and refusal when neither declares it. The
count is unrelated to `len(dspark_target_layer_ids)` (drafter depth versus
tapped main-stack layers) and is never inferred from it. The manifest
provenance records the resolved `n_mtp_layers` and `n_mtp_layers_source`.

The DSpark builder has a second experts mode, `--experts-format iqk
--routed-artifacts <dir>`, that takes the routed projections from
pre-encoded IQ_K conversion artifacts instead of the source checkpoint. The
staging directory holds one raw cell per stage and projection (expert-major
ik-wire rows) plus an `inventory.json` with per-file digests, written by the
converter only after every unit completes; the builder refuses a staging
directory without the inventory and verifies every digest before consuming
a cell. Each expert row is rearranged onto the decode kernels'
`iqk_relayout` under the same two gates the package relayout step runs
(every row unpacked back to wire bytes and compared exactly, plus a
deterministic per-expert row sample decoded through ik's CPU dequantizer
and the relayout reference and compared as fp16 bit patterns), then stored
as one uint8 tensor per stage and projection under
`blocks.N.mlp.switch_mlp.<proj>.iqk_blocks`, shaped
`[num_experts, out_features, row_bytes]`. The manifest row carries the
member (`iqk_codec`), the layout, and the projection geometry per tensor,
so a mixed-member allocation is a manifest fact rather than a schema
change; the consumed staging digests and the gate counts land under the
manifest's provenance. At load the stored rows are not draft-tree parameters:
the loader replaces each stage's `mlp.switch_mlp` with the resident IQ_K switch
in `runtime/deepseek_v4/iqk_experts.py`, splits the stored rows into the
relayout streams the kernels take, verifies every blocks tensor against its
manifest row, and refuses any layout other than `iqk_relayout` by name. DSpark
keeps all of its routed rows resident; the target model's pooled installation
does not wrap the sidecar. The resident switch also remains the reference
implementation for full-capacity target-pool identity checks.

Three manifest kinds exist, all at schema version 1:

- `deepseek_v4_dspark_sidecar`: the DSpark drafter (three draft blocks, the
  Markov head, and the confidence head), one shard per draft stage.
- `deepseek_v4_mtp_sidecar`: the MTP drafter (one vendored decoder block plus
  the fusion projections and norms), one shard. The manifest records the
  chained draft depth cap (block size 3).
- `deepseek_v4_dflash_sidecar`: the DFlash drafter (five dense llama-type
  layers, the `fc` feature projection, the pruned 32000-entry head, and the
  raw d2t/t2d vocabulary tables), one shard. Projections and the head are
  affine 8-bit; the norms and the tables are passthrough. `fc` is affine
  8-bit by default (measured acceptance-neutral and cheaper to ingest);
  the builder's `--fc-format bf16` produces a source-dtype passthrough
  variant. The choice is recorded per tensor and under the manifest's
  `build_options.fc_format`. The shard additionally carries
  sample rows of the dropped verifier embedding under the reserved
  `embed_sample.*` names; the loader compares them against the target
  embedding and logs the delta.

Per-tensor formats come from a four-entry vocabulary; an unknown format fails
closed at load:

| Format | Params | Used for |
|---|---|---|
| `mxfp4` | group 32, bits 4 | routed draft experts: byte-lossless repack from source FP4 (E2M1 packed, UE8M0 per-32 scales) into the MLX mxfp4 layout, with a per-group dequant identity check at build |
| `affine8` | group 32, bits 8 | dense FP8 projections after dequant (attention, shared experts, DSpark `main_proj`, MTP `e_proj`/`h_proj`), and any expert group that failed the mxfp4 identity check (recorded in the manifest) |
| `passthrough` | source dtype, or fp32 where noted | norms and Markov tables (source dtype); router gate weight and bias, hyper-connection parameters, attention sink, and the confidence projection (fp32) |
| `iqk` | `iqk_codec` member, `iqk_relayout` layout, per-tensor `num_experts`/`out_features`/`in_features` | DSpark routed draft experts relaid from IQ_K conversion artifacts, served through the mlx-iqk decode kernels |

No sidecar stores an embedding; the drafter shares the target package's at
load time. DSpark and MTP also share the target language-model head, while
DFlash carries its own pruned draft-vocabulary head.

### The bundled drafter component

`moespresso-ds4-dspark-bundle` writes a new package directory whose files are
hard links to the source package's shards and furniture plus a DSpark
sidecar's shards and manifest, and extends the package manifest with a
declared `drafter` component (`package/deepseek_v4/dspark_bundle.py`). The
component records the identity (path, size, sha256) of every sidecar file, the
sidecar's own artifact id, and the source package's manifest id, so
`moespresso-verify` covers the drafter bytes and the runtime resolves the
drafter from the manifest alone. This is the step that lets a package select a
drafter automatically; a package with no `drafter` key serves plain.

The component is declared `optional` under an all-or-nothing contract: a
distribution of the same package without the drafter files verifies clean and
serves with the absence counted, while a partially present component is
corruption and fails verification. Removing the drafter from a bundled package
is deleting its declared files, never a rebuild. Bytes are never re-encoded;
a link that cannot be created falls back to a copy and the bundle report
records which files copied.

## Why this matters

Every fact above is in the manifest or shard metadata and is declared at
package time. The on-demand verifier confirms content identity and generated
sidecar semantics; the runtime then instantiates the architecture from the
manifest's `config`, maps keys to modules via `expert_layout`, dequantizes via
the declared `format` + `format_params`, checks `required_ops`, and tokenizes
via the declared `rendering_id`. **The engine never guesses, because the format
is fully explicit.**
