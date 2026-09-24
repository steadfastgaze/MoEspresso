# DeepSeek-V4-Flash package recipe

The release package is `DeepSeek-V4-Flash-0731-2.37bpw-MoEspressoV2`. It is
built from the DeepSeek-V4-Flash 0731 checkpoint with IQ_K routed experts, a
q6_K dense side, and a bundled DSpark drafter, and only the MoEspresso engine
loads it. Running the model needs the package alone, not the source checkpoint,
a recipe GGUF, or an importance matrix. The model card carries the download
command for the published repository; verification is part of acquisition:

```bash
uv run --locked moespresso verify <package-dir>
```

The rest of this document describes how packages are built.

## Build routes

Two builders have produced DeepSeek-V4-Flash packages, and they differ in where
the routed-expert bytes come from:

- **Converted IQ_K artifacts** (`moespresso-ds4-iqk-package`). Routed experts
  arrive already encoded in the IQ_K block codecs from a separate conversion
  stage; the builder assembles those bytes and quantizes the dense side
  itself. This route produced the release package. It reads neither a GGUF
  recipe nor an imatrix file.
- **GGUF K-quant recipe** (`moespresso-ds4-kquant-package`). A GGUF supplies
  the tensor-by-tensor codec allocation and, with `--copy-gguf-expert-bytes`,
  the routed-expert bytes themselves.

Every route converges on one `package_plan` before the writer runs, then writes
safetensors shards, package-owned sidecars, and `package_manifest.json`. The
runtime reconstructs the model from that manifest and never reads the source
checkpoint or reparses a GGUF.

## Install the development environment

```bash
make install
```

The default installation includes the runtime and package-builder dependencies.
Verification itself remains a pure integrity path. Routed IQ_K experts decode
through `mlx-iqk` and K-quant tensors through `mlx-kquant`; both are pinned
runtime dependencies, so both are present after `make install`.

## Converted IQ_K artifact route

This is the route that produced the release package. The routed experts are
not encoded here: a conversion stage encodes them, this builder assembles the
bytes, and a second command rearranges them onto the layout the decode kernels
read.

### The staging contract

`--routed-artifacts` points at a directory holding one file per (layer,
projection) cell, named `layer<LL>_<projection>.<member>`, carrying every
expert in index order with no header and no padding, each expert's rows in row
order at the member's own row size. The builder recomputes that arithmetic from
the member's geometry and checks every file's size before it reads a byte, so a
cell converted at a different member or a truncated write fails at open rather
than at serve. `--routed-inventory` takes the conversion's own inventory JSON;
when it is given, every artifact file must reproduce its recorded size and
sha256, which is how the release package was built.

`--allocation` takes the candidate allocation JSON and `--candidate` selects
one record by exact name or unambiguous prefix. The record's
`allocation_map.per_layer` names an IQ_K member for every layer's gate, up, and
down cell. A mixed allocation is the normal case: the member is a per-(layer,
projection) fact through the plan, the bundle metadata, and the manifest, and
nothing downstream reads a package-wide bit width.

### The dense side

`--dense-codec` names the codec for every dense tensor the inventory maps to a
GGUF tensor name, which is exactly the set a K-quant recipe covers; no GGUF
file is read. The remainder keeps the conservative storage the recipe route
also gives it, 8-bit affine or mxfp8 for an fp8 source that ships its own block
scales. The embedding has no such mapping and stays affine, while the head has
one and takes the codec. The default is `q8_0`; the release package uses
`q6_k`. Dense tensors use the K-quant codec registry; IQ_K remains specific to
the routed experts in this builder.

```bash
uv run --locked moespresso-ds4-iqk-package \
  <hf-source-dir> <package-dir> \
  --routed-artifacts <conversion-dir> \
  --allocation <candidates.json> \
  --candidate <candidate-name> \
  --routed-inventory <conversion-dir>/inventory.json \
  --dense-codec q6_k \
  --calibration-capture <capture-dir>
```

`--preflight-only` validates the allocation, the converted artifacts, and the
dense side without writing a package. `--calibration-capture` ranks the
cold-start expert hotlist from route-active calibration counts; without it, or
when the counts do not align with the package's expert index, the builder falls
back to the vendored ranking and never ships a misaligned one. The build writes
the usual furniture: inventory artifact, package plan, shards, manifest,
generated sidecars, tokenizer files, agentic profile, hotlist, and a build
report.

### Relayout, and why it is mandatory

The builder stores the quantizer's own row-major stream, which the manifest
records as `layout: "ik_wire"`. The decode kernels read a k-contiguous
rearrangement of the same bits, `iqk_relayout`, and a package still on the wire
layout is refused by name at install. Only the relayout output serves:

```bash
uv run --locked moespresso-ds4-iqk-relayout <package-dir>
uv run --locked moespresso verify <package-dir>
```

The step rewrites in place, or into `--output` when the package's own footprint
is free again. No encoder runs, no source checkpoint is read, and no
reconstructed value changes; a relayout row is exactly as wide as the wire row
it replaces, so every bundle shape, component offset, and row stride stays
where it was. Two gates run over the rewritten bytes: every row is turned back
into wire bytes and compared with the bytes it came from, and a deterministic
per-expert row sample is decoded twice, the wire bytes through ik's CPU
dequantizer and the rewritten bytes through the relayout reference, compared as
raw fp16 bit patterns. The rewrite is per shard and atomic, and a shard whose
bundles already declare the relayout is skipped, so an interrupted run resumes
from the shards' own recorded layout. `--no-round-trip` drops the every-row
gate, which is the gate that proves these bytes survived the rearrangement.

The relayout writes new plan and manifest ids, so verify afterwards.

### Recorded release build

The release package holds 47 model shards and 1,328 tensors at 84.35 GB
(78.56 GiB), and 90.74 GB (84.51 GiB) including the bundled drafter, 64 files
in total. Routed experts are 77.88 GB as a mix of IQ2_KS on 89 cells and IQ2_K
on 40, averaging 2.2491 bits per weight; everything else is 6.47 GB. The
whole-model rate is 2.37 bits per weight over the served model's 284.335e9
parameters with the drafter excluded from both sides of the ratio. Package
identity and file hashes, rather than the size alone, identify a build.

## GGUF K-quant recipe route

This route builds from three matching inputs:

1. the original Hugging Face safetensors checkpoint;
2. a GGUF file used as the tensor-by-tensor K-quant recipe and, optionally, as
   the routed-expert byte source;
3. a llama.cpp importance matrix used by imatrix-steered codecs.

No public recipe GGUF and imatrix URL is recorded here. The files must describe
the same DeepSeek-V4-Flash model as the source checkpoint.

### Preflight

Run preflight before encoding a full package:

```bash
uv run --locked moespresso-ds4-kquant-package \
  <hf-source-dir> <package-dir> \
  --gguf-recipe <recipe.gguf> \
  --imatrix <imatrix-file> \
  --preflight-only
```

Preflight validates the source inventory, recipe mapping, codec geometry, and
imatrix fit without encoding weights. It writes the recipe report under the
output directory and exits nonzero on a blocking mismatch.

### Byte-faithful build

The byte-faithful build path copies routed-expert wire bytes from the GGUF and
re-encodes dense tensors from the original checkpoint:

```bash
uv run --locked moespresso-ds4-kquant-package \
  <hf-source-dir> <package-dir> \
  --gguf-recipe <recipe.gguf> \
  --imatrix <imatrix-file> \
  --copy-gguf-expert-bytes \
  --kquant-cache-dir <cache-dir> \
  --optimized-kernels-expected
```

`--copy-gguf-expert-bytes` requires the routed expert codecs to match the GGUF
recipe. It preserves the reference quantizer's discrete expert-byte decisions
while retaining the MoEspresso manifest, shard, verification, and runtime path.
Dense tensors continue through the normal package writer.

## K-quant package modes

### Recipe-faithful re-encode

Without `--copy-gguf-expert-bytes`, the default builder follows the GGUF codec
allocation but re-encodes routed expert weights from the source checkpoint.
Some IQ codecs have no GPU encoder, so a complete re-encode can take many hours.
This path is useful when proving the encoder or rebuilding without GGUF bytes;
it is not byte-identical to the source GGUF.

### Fast diagnostic

`--fast-diagnostic` replaces every routed `iq*` target, including gate, up, and
down projections, with `q2_k`. It exists for package and runtime wiring checks.
It is not recipe-faithful and is not quality evidence.

`--force-very-slow-cpu-iquant-encode` may be combined with
`--fast-diagnostic` to keep the IQ codecs. The option intentionally restores
the slow CPU encode and is only for an explicit encoder investigation. It has
no effect outside a fast-diagnostic build.

### Smoke package

`--smoke` is shorthand for `--max-experts-per-layer 1`. A reduced-expert
package can check schema, shard writing, loading, and a short generation. It is
a declared smaller model and cannot provide quality or speed evidence for the
full package.

## Draft-model sidecars

Speculative decoding
([`speculative_decoding.md`](speculative_decoding.md)) loads a drafter from a
standalone sidecar folder. The release provides DSpark and DFlash builders.
Each
resolves source names once at build time and writes safetensors shards plus a
content-hashed manifest that the loader validates fail-closed. No sidecar
stores an embedding; the drafter shares the target package's. DSpark also
shares the target language-model head, while DFlash carries its own pruned
draft-vocabulary head. `--output` takes a path, or a bare name placed under the
Hugging Face hub cache.

The DSpark sidecar reads the `mtp.{0,1,2}` draft tensors from the
DeepSeek-V4-Flash-DSpark checkpoint (three draft blocks, the Markov head, and
the confidence head). This is the drafter the release package bundles:

```bash
uv run --locked moespresso-ds4-dspark-sidecar \
  --source <dspark-snapshot-dir> \
  --output moespresso-ds4-dspark-mxfp4-sidecar
```

`--experts-format iqk --routed-artifacts <staging-dir>` takes the routed draft
projections from pre-encoded IQ_K conversion artifacts instead of the source
checkpoint, under the same staging contract and the same two relayout gates the
package route uses. The default, `mxfp4`, converts the checkpoint's own
experts. The release package bundles a sidecar built the IQ_K way, with its
routed draft experts at the IQ2_K member.

The source tree retains an MTP builder as quarantined research code for a
future compatible checkpoint. It has no installed command in this release.
The current checkpoint cannot produce a valid sidecar for that implementation;
the compatibility limits are in
[`speculative_decoding.md`](speculative_decoding.md).

The DFlash sidecar reads a different checkpoint, the
`RedHatAI/DeepSeek-V4-Flash-speculator.dflash` snapshot, which stores five
dense llama-type layers, the `fc` feature projection, a pruned 32000-entry
head, and the raw d2t/t2d vocabulary tables under final module-path names
rather than `mtp.*` names:

```bash
uv run --locked moespresso-ds4-dflash-sidecar \
  --source <speculator-snapshot-dir> \
  --output moespresso-ds4-dflash-sidecar
```

`--fc-format bf16` stores the `fc` projection as a source-dtype passthrough
instead of the 8-bit affine default; the choice is recorded per tensor and
under the manifest's `build_options`.

The DSpark builder repacks routed draft experts byte-losslessly from source FP4
into the MLX mxfp4 layout. Dense FP8 projections are dequantized and
affine-quantized at 8 bits, and the
precision-sensitive glue (norms, Markov tables, router gate, hyper-connection
parameters, attention sink, confidence projection) stays unquantized. DFlash
has no FP4 experts to repack: its projections and head are affine 8-bit and its
norms and vocabulary tables are passthrough. The recorded mxfp4 DSpark sidecar
is 140 tensors in three shards at about 10 GiB. The manifest kinds and
per-tensor format tables are documented in
[`package_format.md`](package_format.md).

### Bundling a drafter into a package

A sidecar on its own is a folder the runtime must be pointed at. Bundling it
into the package is what lets the drafter be selected automatically:

```bash
uv run --locked moespresso-ds4-dspark-bundle <package-dir> <sidecar-dir> \
  --output <bundled-package-dir>
```

The bundler writes a new package directory whose files are hard links to the
inputs, so no bytes are re-encoded, and extends the package manifest with a
declared `drafter` component carrying every sidecar file's identity. The
component is optional under an all-or-nothing contract, so a distribution
without the drafter files still verifies and serves. Verify the bundled
package, then serve it; the runtime resolves the drafter from the manifest and
engages it when the memory budget allows
([`speculative_decoding.md`](speculative_decoding.md)).

## Encode cache

`--kquant-cache-dir <cache-dir>` stores encoded K-quant payloads by source
content, codec, and relevant imatrix/encode parameters. Repeated builds reuse
unchanged tensors and invalidate only affected cache entries.

The cache is a build accelerator. Cache paths and cache contents are not part
of the package contract and are not written into public provenance.

## Controlled overrides

Both package builders support:

- `--force-format <pattern>=<format>` to override matched package-plan rows;
- `--force-format-dry-run` to write the plan and report without encoding;
- `--allow-unmatched-force` to permit an intentionally unmatched pattern.

Unknown formats fail closed. Unmatched patterns fail closed unless explicitly
allowed. Overrides are recorded in the package plan and manifest, and are
research tools rather than the standard build path.

`--optimized-kernels-expected` records that the package is intended for the
optimized DeepSeek runtime. It does not bypass format, shape, or runtime checks.

## Verify and run

Verify after every build, copy, or move:

```bash
uv run --locked moespresso verify <package-dir>
```

The command checks manifest validity, declared files, sizes, sha256 hashes, and
the tensor keys present in each shard. Exit status 0 is clean; exit status 2
means the package has a blocking integrity problem.

After verification:

```bash
uv run --locked moespresso generate \
  <package-dir> --prompt "Hello" --max-tokens 64

uv run --locked moespresso serve <package-dir>
```

DeepSeek rendering is a runtime-owned contract. The generate and serve commands
map `--thinking off|on|high|max` onto the official encoder modes: `off` renders
chat mode (the default), `on` and `high` render thinking mode, and `max` adds
the official maximum reasoning-effort preamble. The selection is fixed at
startup; per-request render fields stay rejected so the served prefix and
attention rail are stable.

## Quality requirement

A successful build and manifest verification prove package integrity. Model
quality requires the gates in
[`deepseek_v4_quality.md`](deepseek_v4_quality.md) before promoting a package.
Fast-diagnostic and smoke packages must never be substituted for that evidence.
