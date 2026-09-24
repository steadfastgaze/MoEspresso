<img src="docs/assets/hero.webp" alt="MoEspresso" width="100%">

# MoEspresso

MoEspresso runs large Mixture-of-Experts language models at practical speed on
Apple Silicon, including Macs whose memory is much smaller than the model
package.

MoEspresso 3 runs **Qwen3.8-Flash-Next on a 2021 M1 Max with 32 GB of unified
memory at roughly 12–15 decode tokens per second**. It retains all 512 routed
experts and the original PLE memory while serving with a 128K context limit.

On a [randomly selected 48-question set](#quality-on-the-bounded-path), the
Cache-Prior policy used for 32 GB serving scored **84.3%**, ahead of Claude Opus
4.8 xhigh at **80.3%** on the same questions.

| MoEspresso 3 result | Value |
|---|---|
| Model | Qwen3.8-Flash-Next, 125B MoE, 6B active per token |
| Package payload | 145.75 GB, including 102.40 GB of original PLE data |
| Routed experts | All 512 retained |
| 2021 M1 Max, 24-core GPU, 32 GB | 12.96–15.85 decode tok/s across the physical grid, 15.46 tok/s on the Cache-Prior 2/2 release cell |
| Frozen generated-answer comparison | 84.3% with Cache-Prior 2/2 at capacity 223 |
| Default served context | 131,072 tokens |

The physical speed grid used the complete 48-layer package at capacity 223, a
24 GB planner ceiling, a 43-token prompt, 32 generated tokens, greedy decoding,
and disk KV disabled.

## MoEspresso 3 package

The package is
[Qwen3.8-Flash-Next-MoEspressoV3](https://huggingface.co/steadfastgaze/Qwen3.8-Flash-Next-MoEspressoV3).
It is the first package built for MoEspresso 3's Qwen-specific storage,
attention-cache, and bounded-routing stack.

The model shards occupy 43.35 GB. Most routed projections use IQ2_K, a format
originating in ik_llama.cpp and implemented here in Metal, while six early-layer
projection groups are promoted to IQ3_K. Dense tensors stay at the
higher precision assigned during package construction, while PLE remains
package-backed on SSD. The package contains the target text model without its
vision stack or a separate MTP sidecar.

## Install and run

MoEspresso 3.0.0 requires an arm64 Apple Silicon Mac running macOS 26.2
(Tahoe) or later.

Install the engine and the Hugging Face CLI with Homebrew:

```bash
brew install steadfastgaze/tap/moespresso
brew install hf
```

Download the package into an explicit directory:

```bash
hf download steadfastgaze/Qwen3.8-Flash-Next-MoEspressoV3 \
  --local-dir ./models/qwen3.8-flash-next
```

You can verify its manifest, file identities, tensor keys, and sidecars:

```bash
moespresso verify ./models/qwen3.8-flash-next
```

Start the OpenAI-compatible server:

```bash
moespresso serve ./models/qwen3.8-flash-next
```

The server performs a short warmup, then exposes
`POST /v1/chat/completions` and `GET /health` on `127.0.0.1:8080`. Qwen
thinking is enabled at medium effort by default. Requests may select `low`,
`medium`, or `xhigh` through the OpenAI-compatible `reasoning_effort` field.

For one generation without the HTTP server:

```bash
moespresso generate ./models/qwen3.8-flash-next \
  --prompt "Explain why B-trees work well on SSDs." \
  --max-tokens 512
```

### Measure decode speed

With the server running, use the short built-in speed check:

```bash
moespresso speed
```

It reports topic-switch, same-topic, and overall decode rates together with
the hardware, context, expert capacity, memory plan, and Cache-Prior settings.
It also warns when other applications have reduced the expert pool enough to
make the result unrepresentative. See [checking decode speed](docs/diagnostics.md)
for the fixed prompt sequence and timing formula.

### Use it with OpenCode

The server works with OpenAI-compatible clients. This shell function configures
a local provider for OpenCode without changing its global configuration:

```bash
moecode() {
  OPENCODE_CONFIG_CONTENT='{
    "provider": {
      "moespresso": {
        "npm": "@ai-sdk/openai-compatible",
        "name": "MoEspresso (local)",
        "options": { "baseURL": "http://127.0.0.1:8080/v1" },
        "models": {
          "Qwen3.8-Flash-Next": {
            "name": "Qwen3.8-Flash-Next @ MoEspresso",
            "temperature": true,
            "interleaved": "reasoning_content",
            "limit": { "context": 131072, "output": 32768 }
          }
        }
      }
    },
    "model": "moespresso/Qwen3.8-Flash-Next",
    "small_model": "moespresso/Qwen3.8-Flash-Next"
  }' opencode "$@"
}

moecode
```

Reasoning is returned in `reasoning_content`, so clients can preserve it across
turns. Tool calls use the model's native dialect and are translated to the
OpenAI-compatible response shape.

### Main serving controls

- `--max-memory-gb` sets the startup capacity planner's ceiling. It changes
  expert-pool geometry and is not a process RSS limit.
- `--max-context-tokens` selects a context limit up to the package's declared
  262,144-token architecture limit. The default is 128K.
- `--min-resident-experts` refuses startup when the resolved per-layer pool is
  smaller than the requested capacity.
- `--cache-routing off` disables the bounded Cache-Prior policy. `auto`, the
  default, enables Cache-Prior only when the pool cannot keep every routed
  expert resident.
- `--thinking off|on` selects the Qwen template mode. Thinking defaults to on,
  with medium effort.
- `--prompt-cache-size` and `--prompt-cache-bytes` bound retained in-memory
  prompt state.

## Quality on the bounded path

The release comparison uses four questions from each of two tasks in each of six
public LiveBench categories: coding, data analysis, instruction following,
language, math, and reasoning. A fixed salted SHA-256 rule derives the selection
from pinned datasets, and the official task scorers grade every complete
answer. The official LiveBench leaderboard uses a different generation
protocol.

| Model and effort | Macro score |
|---|---:|
| Hosted Qwen3.8 Flash, xhigh | 90.7% |
| GPT-6 Sol, medium | 89.2% |
| Claude Opus 5.5, low | 87.6% |
| **MoEspresso, IQ_K routed experts, Cache-Prior 2/2, capacity 223, medium** | **84.3%** |
| GPT-6 Luna, xhigh | 81.4% |
| Claude Opus 4.8, xhigh | 80.3% |

The MoEspresso arm used greedy generation, medium reasoning effort, all 512
experts retained in the package, 223 resident slots per layer, and the same
Cache-Prior 2/2 policy used by the 32 GB serving configuration. Hosted Qwen
used the fixed sampling controls, while the GPT and Claude endpoints required
their provider defaults. All hosted rows ran through OpenRouter. None of the 48
local answers reached the output ceiling. The complete selection procedure,
runner, isolated graders, and scoring code are in the
[benchmark reproduction kit](docs/benchmark_reproduction/).

## Package generations

Qwen3.8-Flash-Next is the only package built specifically for MoEspresso 3.
Earlier releases remain available and supported by the 3.x line:

| Generation | Package | Status |
|---|---|---|
| MoEspresso 3 | [Qwen3.8-Flash-Next-MoEspressoV3](https://huggingface.co/steadfastgaze/Qwen3.8-Flash-Next-MoEspressoV3) | Current development focus |
| MoEspresso 2 | [DeepSeek-V4-Flash-0731-2.37bpw-MoEspressoV2](https://huggingface.co/steadfastgaze/DeepSeek-V4-Flash-0731-2.37bpw-MoEspressoV2) | Legacy, supported in 3.x |
| MoEspresso 2 | [DeepSeek-V4-Flash-0731-Coder-56.8GB-MoEspressoV2](https://huggingface.co/steadfastgaze/DeepSeek-V4-Flash-0731-Coder-56.8GB-MoEspressoV2) | Legacy, supported in 3.x |
| MoEspresso 1 | [DeepSeek-V4-Flash-IQ2_XXS-MoEspresso](https://huggingface.co/steadfastgaze/DeepSeek-V4-Flash-IQ2_XXS-MoEspresso) | Legacy, supported in 3.x |
| MoEspresso 1 | [Ornith-1.0-35B-Q4_K_M-MoEspresso](https://huggingface.co/steadfastgaze/Ornith-1.0-35B-Q4_K_M-MoEspresso) | Legacy, supported in 3.x |

MoEspresso 4 will deprecate the legacy packages. The project has one
maintainer, so narrowing the package set leaves time for deeper Qwen work,
further inference improvements, new hardware profiles, and the next model
family. Qwen3.8-Flash-Next performs better for the coding and agentic workloads
targeted here, and MoEspresso 3 makes it fast enough for interactive work on a
32 GB M1 Max.

The DeepSeek and Ornith packages released for MoEspresso 1 use K-quant routed
experts. The DeepSeek packages released for MoEspresso 2 use IQ_K routed experts.

Measurements for the legacy releases remain in
[DeepSeek quality](docs/deepseek_v4_quality.md),
[DeepSeek performance](docs/deepseek_v4_speed.md), and the
[documentation map](docs/README.md).

AMD support is planned for MoEspresso 5, alongside even better Apple Silicon
support. Hardware vendors interested in seeing these models pushed on their
systems are welcome to get in touch.

## How 145.75 GB runs on 32 GB

Qwen3.8-Flash-Next activates ten of its 512 routed experts in each MoE layer.
MoEspresso keeps dense weights, routers, norms, shared experts, attention state,
and a bounded set of routed experts in unified memory. Missing expert rows are
loaded from SSD into persistent per-layer pools as routing selects them. Each
stored row contains the gate, up, and down projections contiguously, so one
miss needs one read.

Full residency and SSD-backed execution use the same pooled graph. The startup
planner first reserves memory for the selected context, resident tensors,
workspace, and safety headroom, then gives the remaining budget to expert
slots. On the measured 32 GB configuration this resolved to 223 of the 512
experts per layer.

### Cache-Prior routing

Bounded serving uses a policy inspired by
[Cache-Prior](https://arxiv.org/html/2412.00099v2). During ordinary
single-token decode, resident experts receive a factor-two preference while the
two strongest routes from the model's original selection remain protected. The
chosen experts keep their original router probabilities as contribution
weights.

Prefill is never biased, and layers 0 and 1 retain original routing during
decode. Every expert remains available on SSD, so a sufficiently strong
nonresident route still loads and runs. Fully resident execution stays
unbiased.

This policy changes model output because it can change the selected expert set.
That is why the 2/2 configuration has its own generated-answer measurement.
Read [Cache-Prior routing](docs/cache_prior.md) for the algorithm, its relation
to REAP, and the expert-pool eviction policy.

### PLE and KVarN

The model's 51B-parameter PLE memory remains in its original BF16 form. Its
102.40 GB payload stays on SSD, and the runtime performs selected-row reads
instead of placing the complete table in unified memory.

Qwen attention uses a KVarN K4/V4 cache with an exact BF16 sink and recent
suffix. The hybrid prompt state also includes Gated DeltaNet recurrence and PLE
history, and all components advance to the same committed token frontier.
Their bounded memory use leaves substantially more of the 32 GB budget
available for resident expert slots at 128K. The full architecture and state
contract are in
[Qwen3.8 architecture and serving](docs/qwen4.md).

## Prefix reuse and disk KV

The in-memory prompt cache reuses the longest exact token prefix available in
the current process. MoEspresso also enables a disk checkpoint tier by default,
which lets a restarted server or a new session restore a shared prompt prefix
and prefill only the suffix. This is especially useful for agent clients with a
large, stable system prompt and tool description.

The default store lives below `~/.cache/moespresso/disk_kv`, has an 8 GiB
per-package budget, writes 1024-token frontiers through the first 16K tokens,
and evicts by LRU. Every restore is scoped to the package, renderer, and cache
policy. Invalid or mismatched data fails closed to ordinary prefill.

Disable the disk tier for controlled performance measurements:

```bash
MOESPRESSO_DISK_KV=off \
  moespresso serve ./models/qwen3.8-flash-next
```

See [disk KV](docs/disk_kv.md) for the checkpoint and recovery contract.

## The package contract

A MoEspresso package records the decisions required to reproduce its runtime
graph. The manifest declares the architecture, tensor formats, required backend
operations, tokenizer and renderer identity, content-addressed files, and the
provenance chain leading to the package plan.

Routed layers store expert bundles as `uint8 [n_experts, row_bytes]` tensors in
ordinary safetensors shards. Expert row `e` contains that expert's complete
gate, up, and down payload in the layout consumed by the serving kernels.
Shard metadata records the offsets, shapes, codecs, and component order needed
to locate a row without reading unrelated weight data.

`moespresso verify PACKAGE` checks this contract before use. Verification stays
outside the serving hot path, so startup builds the model from declared package
facts without re-reading source checkpoints or guessing tensor conventions.
Read [package format](docs/package_format.md) for the complete schema.

## Correctness and reproducibility

Plausible text is not a correctness test. Changes to quantized math, attention,
cache state, routing, and package loading are checked against reference tokens
or logits while exercising the intended runtime path. Performance changes use
matched artifacts and explicit A/B conditions.

The public repository includes family-specific gates, shared package checks,
and the generated-answer reproduction kit. Start with the
[correctness ladder](docs/correctness_ladder.md) and
[optimization methodology](docs/optimization_methodology.md) when changing
runtime behavior.

## Documentation

- [Documentation map](docs/README.md): model, package, runtime, quality, and
  benchmark references.
- [Developer guide](DEVGUIDE.md): lifecycle, source map, entry points, and test
  commands.
- [Contributor guide](AGENTS.md): repository rules and runtime invariants.
- [Qwen3.8 architecture](docs/qwen4.md): PLE storage, KVarN state, generation,
  and prompt reuse.
- [Cache-Prior routing](docs/cache_prior.md): decode-time resident preference,
  its REAP analogy, and expert eviction.
- [SSD streaming](docs/ssd_streaming.md): capacity planning, expert pools, and
  direct reads.
- [Disk KV](docs/disk_kv.md): restart-warm and cross-session prefix reuse.
- [Package format](docs/package_format.md): manifests, shards, tensor formats,
  and integrity declarations.
- [Tool calls](docs/tool_calls.md): request dialects, streaming, and repair.

## Acknowledgements

MoEspresso builds on a large body of open work. The complete per-file record is
in `THIRD-PARTY-NOTICES`.

- The [Qwen team](https://huggingface.co/Qwen) trained and released
  [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), whose
  text model and PLE tables form the MoEspresso 3 package.
- <a id="mlx-iqk-acknowledgement"></a>[Iwan Kawrakow](https://github.com/ikawrakow)
  designed the IQ_K formats used by the routed experts. Their reference
  encoders and CPU dequantizers are published in
  [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp). MoEspresso's Metal
  execution is implemented in
  [mlx-iqk](https://github.com/steadfastgaze/mlx-iqk), with bit-exact
  reconstruction against those reference dequantizers as its correctness
  target.
- [Georgi Gerganov](https://github.com/ggerganov),
  [ggml](https://github.com/ggml-org/ggml), and
  [llama.cpp](https://github.com/ggml-org/llama.cpp) established GGUF, the
  block-quantization ecosystem, and a reference engine used throughout this
  work.
- <a id="mlx-kquant-acknowledgement"></a>[Asher Feldman](https://github.com/asher)
  created the original
  [mlx-kquant](https://github.com/asher/mlx-kquant) integration that brought
  K-quant wire formats to MLX. The release uses the
  [steadfastgaze fork](https://github.com/steadfastgaze/mlx-kquant), which adds
  MoEspresso's model-specific K-quant kernels.
- [Bartowski](https://github.com/bartowski1182)'s `calibration_datav5` corpus,
  which in turn credits Dampf, Kalomaze, and edaddario, is one component of the
  DeepSeek package calibration mix. Published GGUF quantizations also provide
  external quality references.
- [turboderp](https://github.com/turboderp-org)'s
  [exllamav3](https://github.com/turboderp-org/exllamav3) contributes technical
  calibration and held-out probe data.
- [froggeric](https://huggingface.co/froggeric)'s
  Qwen-Fixed-Chat-Templates provides the base for the vendored Qwen 3.5/3.6
  template used by the Ornith adapter.
- [Jinho Jang](https://github.com/jjang-ai)'s
  [JANG](https://github.com/jjang-ai/jangq) is the source for several adapted
  kernels recorded in `THIRD-PARTY-NOTICES`.
- [antirez](https://github.com/antirez)'s
  [DwarfStar](https://github.com/antirez/ds4) provided a narrow Qwen and
  DeepSeek reference engine whose quality discipline helped expose subtle
  inference errors.
- Apple and the [MLX](https://github.com/ml-explore/mlx) and
  [mlx-lm](https://github.com/ml-explore/mlx-lm) communities provide the array
  framework, graph runtime, model components, and Apple Silicon ecosystem on
  which MoEspresso runs.

## License

MoEspresso is dual-licensed under Apache 2.0 (`LICENSE-APACHE-2.0`) or MIT
(`LICENSE-MIT`), at your option.

`THIRD-PARTY-NOTICES` records the attribution and upstream revision for
third-party code that ships in this repository. The wheel and source
distribution carry it with both license files.
