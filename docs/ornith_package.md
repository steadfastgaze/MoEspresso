# Ornith package status

The proven Ornith 1.0 35B package is published at
[`steadfastgaze/Ornith-1.0-35B-Q4_K_M-MoEspresso`](https://huggingface.co/steadfastgaze/Ornith-1.0-35B-Q4_K_M-MoEspresso).
Its routed experts use the same SSD-streaming bundle contract as
DeepSeek-V4-Flash. This repository does not yet provide a turnkey rebuild from
the original model's published Hugging Face checkpoint because the public
source adapter is incomplete.

Do not infer a build command from an internal architecture-family entry point.
The public source-to-package path is incomplete until the expert-layout boundary
described below is handled in-tree.

## Proven package and runtime

The recorded package is a byte-faithful Q4_K_M-class artifact:

- about 19.91 GiB;
- 40 layers, including 10 full-attention layers and 30 recurrent/linear-attention
  layers;
- 256 routed experts with top-8 routing plus a shared expert;
- q8 affine KV for the full-attention layers;
- a declared context limit of 262,144 tokens;
- package-owned tokenizer, chat template, and agentic profile sidecar;
- manifest-declared K-quant tensors installed through `mlx-kquant`;
- one contiguous expert bundle per routed layer, indexed by expert row for
  resident or bounded SSD-backed serving.

The package has been exercised resident, through bounded expert residency, with
the model-specific quality gate, and through long cumulative sessions. See
[`ornith_quality.md`](ornith_quality.md) and
[`ornith_speed.md`](ornith_speed.md).

Install the Hugging Face CLI with either Homebrew or `uv`:

```bash
brew install hf
```

```bash
uv tool install hf
```

Then download the prebuilt package and verify it before loading any weights:

```bash
hf download steadfastgaze/Ornith-1.0-35B-Q4_K_M-MoEspresso \
  --local-dir <package-dir>
moespresso-verify <package-dir>
```

## Why rebuilding from the published checkpoint is blocked

The published checkpoint stores three separate 2D tensors for every routed
expert: `gate_proj`, `up_proj`, and `down_proj`. With 256 experts, each layer
therefore exposes 768 routed-expert source tensors.

The current Qwen source inventory and recipe mapper instead recognize two
stacked 3D source tensors per layer:

```text
experts.gate_up_proj.weight  [experts, 2 * intermediate_features, hidden_features]
experts.down_proj.weight     [experts, hidden_features, intermediate_features]
```

Together these sources map to three logical recipe projections: gate and up are
split from `gate_up_proj`, while down comes from `down_proj`. The mapper does
not currently assemble or stream from the checkpoint's per-expert names.

This is only a source-ingestion gap. The package writer already converges both
Ornith and DeepSeek-V4-Flash onto one per-layer bundle with logical shape
`[experts, row_bytes]`. The expert index and pooled runtime consume that bundle
directly for resident or SSD-backed serving.

A one-off pre-stacked source view was used to build the proven package. That
view was an input aid; it is not part of the package and is not required by the
runtime. No supported source adapter or pre-stack command ships in this tree.
Consequently:

- there is no documented public command that consumes the published checkpoint
  directly;
- the current Qwen builder must not be presented as a turnkey command for the
  published Ornith checkpoint;
- the published prebuilt package can be downloaded, verified, and served
  normally;
- documenting a reproducible source rebuild must wait for an in-tree source
  adapter.

## What the source adapter must preserve

A public source adapter must provide a lossless path into the existing package
writer. It must:

1. discover every expected expert and projection from checkpoint headers;
2. order experts deterministically by layer, projection, and expert id;
3. present the per-expert matrices to the encoder in that order, either by
   streaming them directly as DeepSeek-V4-Flash does or by constructing the
   current stacked source view;
4. preserve dtype and tensor values without quantizing or transposing them;
5. fail closed on missing, duplicate, or inconsistent expert shapes;
6. record enough source identity to make the resulting package reproducible;
7. pass preflight, manifest verification, and the Ornith quality gate.

The adapter should be integrated into the public builder. Direct per-expert
streaming is preferable because it avoids rewriting a second full checkpoint
solely to satisfy the current inventory shape.

## Verify an existing package

Run the integrity gate after receiving, copying, or moving a package:

```bash
uv run --locked moespresso-verify <package-dir>
```

The command checks the package manifest, every declared file's path, size and
sha256, and the declared tensor keys in each shard. Exit status 0 is clean; exit
status 2 means a blocking integrity problem was found.

Verification proves package integrity. It does not establish model quality or
that the package matches the recorded byte-faithful artifact.

## Generate and serve

```bash
uv run --locked moespresso-generate \
  <package-dir> --prompt "Hello" --max-tokens 64 --thinking off

uv run --locked moespresso-serve \
  <package-dir> --thinking off
```

The server warms one isolated generation before announcing readiness. It then
serves `POST /v1/chat/completions` and `GET /health`; the one-shot command uses
the same manifest-driven load and render path.

Thinking is controlled through the packaged template for Ornith and may be set
with `--thinking on|off` (`high` is an alias of `on`; `max` is a DeepSeek-V4
reasoning-effort level and refuses loudly here). The recorded quality and
performance evidence in this repository uses thinking off unless a document
states otherwise.

## Memory policy

The same package can run fully resident or with a bounded routed-expert pool.
Pass `--max-memory-gb <budget>` to `moespresso-serve` or
`moespresso-generate` to set the startup capacity-planner ceiling. Capacity is
derived from the package's actual expert-row geometry after fixed runtime and
KV/activation allowances. This selects pool geometry; it is not an RSS limit,
and the pool does not shrink as context grows.

The 32 GB operating point in the public record was reproduced by a capacity cap
on a 128 GB host. It proves bounded-memory behavior and fit, but not the SSD
stall time of a physical 32 GB machine.

## Restart-warm sessions

The opt-in disk KV tier works with Ornith's hybrid cache, including recurrent
state. It restores only exact token-prefix checkpoints and fails closed to cold
prefill on mismatch. Configuration and safety evidence live in
[`disk_kv.md`](disk_kv.md).

## Promotion checklist

Before an Ornith package is described as supported:

- `moespresso-verify` passes after the final copy;
- the manifest id and package file hashes are recorded;
- the full gate in [`ornith_quality.md`](ornith_quality.md) passes;
- resident and intended streamed modes are tested;
- the MLX wheel and runtime dependency versions are recorded;
- no private benchmark questions, answers, or local filesystem paths are
  included in the package or public evidence.
