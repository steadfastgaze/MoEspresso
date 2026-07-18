# DeepSeek-V4-Flash package recipe

The release package is published at
[`steadfastgaze/DeepSeek-V4-Flash-IQ2_XXS-MoEspresso`](https://huggingface.co/steadfastgaze/DeepSeek-V4-Flash-IQ2_XXS-MoEspresso).
Users who want to run the model do not need the source checkpoint, recipe GGUF,
or importance matrix. Install the Hugging Face CLI with either Homebrew or
`uv`:

```bash
brew install hf
```

```bash
uv tool install hf
```

Then download and verify the package:

```bash
hf download steadfastgaze/DeepSeek-V4-Flash-IQ2_XXS-MoEspresso \
  --local-dir <package-dir>
moespresso-verify <package-dir>
```

The rest of this document describes how the package itself was built.

MoEspresso builds a DeepSeek-V4-Flash package from three matching inputs:

1. the original Hugging Face safetensors checkpoint;
2. a GGUF file used as the tensor-by-tensor K-quant recipe and, optionally, as
   the routed-expert byte source;
3. a llama.cpp importance matrix used by imatrix-steered codecs.

The GGUF supplies build-time tensors. Runtime reads the generated MoEspresso
package. The builder writes safetensors shards, package-owned sidecars, and
`package_manifest.json`. The runtime reconstructs the model from that manifest
and never reads the source
checkpoint or reparses the GGUF.

No public recipe GGUF and imatrix URL is recorded here. The files must describe
the same DeepSeek-V4-Flash model as the source checkpoint.

## Install the development environment

```bash
make install
```

The default installation includes the runtime and package-builder dependencies.
Verification itself remains a pure integrity path.

## Preflight

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

## Byte-faithful build

The release-quality build path copies routed-expert wire bytes from the GGUF and
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

The recorded byte-faithful package has 47 shards and is about 80.0 GiB. Package
identity and file hashes, rather than the size alone, identify a build.

## Package modes

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

## Encode cache

`--kquant-cache-dir <cache-dir>` stores encoded K-quant payloads by source
content, codec, and relevant imatrix/encode parameters. Repeated builds reuse
unchanged tensors and invalidate only affected cache entries.

The cache is a build accelerator. Cache paths and cache contents are not part
of the package contract and are not written into public provenance.

## Controlled overrides

The builder supports:

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
uv run --locked moespresso-verify <package-dir>
```

The command checks manifest validity, declared files, sizes, sha256 hashes, and
the tensor keys present in each shard. Exit status 0 is clean; exit status 2
means the package has a blocking integrity problem.

After verification:

```bash
uv run --locked moespresso-generate \
  <package-dir> --prompt "Hello" --max-tokens 64

uv run --locked moespresso-serve <package-dir>
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
