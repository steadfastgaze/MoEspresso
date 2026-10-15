<img src="docs/assets/hero.webp" alt="MoEspresso" width="100%">

# MoEspresso

MoEspresso runs large Mixture-of-Experts language models at practical speed on
Apple Silicon, including Macs whose memory is much smaller than the model
package.

It keeps frequently used experts in unified memory and loads the others from
SSD as needed. All experts remain available. The runtime is written in Rust
with native Metal kernels and provides local generation and an OpenAI-compatible
server for chat, reasoning and coding agents.

With Qwen3.8-Flash-Next, a 145.75 GB package runs on a 2021 M1 Max with 32 GB
of unified memory, retaining all 512 routed experts and the original PLE
memory while serving with a 128K context limit.

## Model package

The supported package is
[Qwen3.8-Flash-Next-MoEspressoV3](https://huggingface.co/steadfastgaze/Qwen3.8-Flash-Next-MoEspressoV3).
An existing download works with MoEspresso 4.0; it does not need conversion.

| Package | Value |
| --- | --- |
| Model | Qwen3.8-Flash-Next, 125B MoE, 6B active per token |
| Model shards | 43.35 GB |
| Original BF16 PLE memory | 102.40 GB |
| Complete payload | 145.75 GB |
| Routed experts | All 512 retained |
| Default served context | 131,072 tokens |

Most routed projections use IQ2_K, a format originating in ik_llama.cpp and
implemented here in Metal. The six projection groups in the first two layers
use IQ3_K. Dense tensors keep their assigned higher precision, while PLE stays
on SSD. The package contains the text model without its vision stack or a
separate MTP sidecar.

Older DeepSeek and Ornith packages are no longer supported in MoEspresso 4.0.
To use them, install MoEspresso 3 instead:

```sh
brew install steadfastgaze/tap/moespresso@3
```

## Install and run

Use an Apple Silicon Mac with Metal support and about 146 GB of SSD space for
the model package. Install the engine and the Hugging Face CLI with Homebrew:

```sh
brew install steadfastgaze/tap/moespresso
brew install hf
```

Download the package into the `Models` folder in your home directory:

```sh
hf download steadfastgaze/Qwen3.8-Flash-Next-MoEspressoV3 \
  --local-dir "$HOME/Models/qwen3.8-flash-next"
```

Start the OpenAI-compatible server:

```sh
moespresso serve "$HOME/Models/qwen3.8-flash-next"
```

The server exposes `POST /v1/chat/completions` and `GET /health` on
`127.0.0.1:8080`. Thinking is enabled at medium effort by default. Requests can
select `low`, `medium` or `xhigh` through `reasoning_effort`. The server runs
one request at a time and keeps conversation state for subsequent turns.
It has no authentication; keep the listener on loopback.

For one generation without the HTTP server:

```sh
moespresso generate "$HOME/Models/qwen3.8-flash-next" \
  --prompt "Explain why B-trees work well on SSDs." \
  --max-tokens 512
```

### Measure speed

With the server running, use the built-in speed check:

```sh
moespresso speed
```

Each topic result prints as its request finishes, with prefill and decode
rates, first-token latency and cache reuse. A separate approximately 4K-token
prefill follows, then an interactive prompt offers a longer 32K test.
The report includes the server's capacity, context, routing and cache settings.
See [speed measurement](docs/speed_comparison.md) for timing definitions and
repeatable comparisons.

### Use it with OpenCode

With the server running and [OpenCode](https://opencode.ai/docs/) installed,
paste this function into Bash or zsh. It selects the local provider without
changing OpenCode's global configuration:

```sh
moecode() {
  OPENCODE_DISABLE_MODELS_FETCH=1 \
  OPENCODE_CONFIG_CONTENT='{
    "enabled_providers": ["moespresso"],
    "provider": {
      "moespresso": {
        "npm": "@ai-sdk/openai-compatible",
        "name": "MoEspresso (local)",
        "options": {
          "baseURL": "http://127.0.0.1:8080/v1",
          "apiKey": "local-placeholder"
        },
        "models": {
          "Qwen3.8-Flash-Next": {
            "name": "Qwen3.8-Flash-Next @ MoEspresso",
            "tool_call": true,
            "reasoning": true,
            "temperature": true,
            "interleaved": "reasoning_content",
            "limit": { "context": 131072, "output": 32768 },
            "options": {
              "reasoningEffort": "medium",
              "chat_template_kwargs": {
                "enable_thinking": true,
                "preserve_thinking": true
              }
            }
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

Run `moecode` from the project directory you want to work on. Reasoning is
returned in `reasoning_content`, and native tool calls are translated to the
OpenAI-compatible response format. The API key is a local placeholder.
Keep OpenCode's context limit at or below the server's `--max-context`.
See [client setup](docs/clients.md) for persistent configuration, thinking
variants and streaming requests.

### Main serving controls

- `--max-memory-gb GIB` caps the automatic memory budget, including the adaptive
  operating ceiling. It is not a process RSS limit.
- `--max-context N` selects the context limit. The default is 131,072 tokens;
  an explicit limit must fit the package and memory budget.
- `--no-adaptive-memory` keeps automatically selected expert capacity fixed.
- `--cache-routing off` disables Cache-Prior. The default is `prefer-resident`;
  fully resident execution stays unbiased.
- `--thinking off` disables thinking by default; requests can override it.
- `--prompt-cache-gb GIB` bounds retained in-memory conversation state. The
  default is 1 GiB; zero disables memory retention.
- `--no-disk-cache` disables disk reads and writes without deleting saved state.
- `--disk-cache-mode full` also saves completed prompts and generated state.
  The default `prefix` mode writes only prefill anchors within the first 16K tokens.
- `--disk-cache-dir PATH` and `--disk-cache-gb GIB` set disk-cache location and
  size. The default retention budget is 8 GiB.

Run `moespresso serve --help`, `moespresso generate --help` or
`moespresso speed --help` for the complete options.

## Quality

The generated-answer comparison uses a frozen set of 48 questions: four from
each of two tasks in each of six public LiveBench categories. The categories
are coding, data analysis, instruction following, language, math and reasoning.
Answers are scored with the official task scorers. This is a custom comparison,
not the official LiveBench leaderboard.

| Model and effort | Macro score |
| --- | ---: |
| Hosted Qwen3.8 Flash, xhigh | 90.7% |
| GPT-6 Sol, medium | 89.2% |
| Claude Opus 5.5, low | 87.6% |
| GPT-6 Luna, xhigh | 81.4% |
| Claude Opus 4.8, xhigh | 80.3% |
| MoEspresso, Cache-Prior 2/2, 223 resident slots, medium | 80.2% |
| MoEspresso, fully resident, unbiased routing, medium | 79.4% |

Both native configurations use the same IQ_K package, greedy generation and
medium thinking. All 48 answers in each configuration stopped naturally,
without reaching the output cap. Hosted scores are retained from the earlier
same-question comparison through OpenRouter; hosted Qwen used fixed sampling
controls, while GPT and Claude used their provider defaults.

The 2/2 score is lower than the earlier release's 84.3%. On this small panel,
the change is not considered a material quality regression: 2/2 remains above
fully resident, unbiased execution, which is the target reference.

## How 145.75 GB runs on 32 GB

Qwen3.8-Flash-Next activates ten of its 512 routed experts in each MoE layer.
MoEspresso keeps dense weights, routers, norms, shared experts, attention state
and a bounded set of routed experts in unified memory. Missing expert rows are
loaded from SSD into per-layer pools as routing selects them.

The startup planner reserves memory for the context, resident tensors,
workspaces, conversation caches and safety headroom, then assigns the remaining
budget to expert slots. During serving, adaptive memory can grow the pools
beyond their starting size during healthy serving or idle periods, or release
slots under memory pressure and sustained active VM churn. Prepared slots are
populated on demand and validated through successful inference. Resizing does
not invalidate conversation caches.
See [adaptive memory](docs/adaptive_memory.md) for the policy and its limits.

### Cache-Prior routing

Bounded serving uses a policy inspired by
[Cache-Prior](https://arxiv.org/html/2412.00099v2). During single-token decode,
resident experts receive a factor-two ranking preference while the two
strongest routes from the model's original selection remain protected.
Contribution weights come from the original router probabilities, normalized
over the selected experts.

Prefill is never biased, and layers 0 and 1 retain original routing during
decode. Every expert remains available on SSD, so a sufficiently strong
nonresident route still loads and runs. Fully resident execution stays unbiased.

The policy can change the selected experts and the model's output, so the 2/2
configuration has its own quality measurement. Read
[Cache-Prior routing](docs/cache_prior.md) for the algorithm and eviction policy.

### PLE and KVarN

The model's 51B-parameter PLE memory remains in its original BF16 form. Its
102.40 GB payload stays on SSD, and the runtime reads selected rows rather than
loading the complete table into unified memory.

Attention uses a KVarN K4/V4 cache with an exact BF16 sink and recent suffix.
Conversation state also includes Gated DeltaNet recurrence and PLE history.
All components advance to the same committed token frontier and are restored
together for prefix reuse.

## Prefix reuse and disk cache

The memory cache reuses the longest compatible exact token prefix, including
consumed generated tokens and short conversations. Both cache tiers are enabled
by default. Disk caching saves shallow prefill prefixes within the first 16K
tokens, allowing common harness instructions and tools to be reused after a
restart without saving a growing snapshot at every turn.

Use `--disk-cache-mode full` for deeper conversation persistence, including
short prompts and generated state. It adds blocking writes before the first
token and after generation. In default `prefix` mode, a restart or RAM-cache
eviction requires reprocessing the conversation after its saved prefix. This
also applies when a conversation snapshot exceeds `--prompt-cache-gb`.
Neither disk mode limits conversation length.

The default disk store is below `~/.cache/moespresso/v4/disk_kv`, or under
`XDG_CACHE_HOME` when set. Its 8 GiB retention budget is shared across managed
cache roots. Invalid or incompatible checkpoints fall back to another valid
prefix or ordinary prefill.

For uncached speed measurements, disable both tiers:

```sh
moespresso serve "$HOME/Models/qwen3.8-flash-next" \
  --no-disk-cache --prompt-cache-gb 0
```

See [disk checkpoints](docs/disk_checkpoints.md) for storage, retention and
recovery behavior.

## The package contract

A MoEspresso package records the architecture, tensor formats, byte layouts,
tokenizer assets and chat template required by the runtime. Routed layers
store expert bundles in safetensors shards; each expert row contains its gate,
up and down projection payloads in the layout consumed by the Metal kernels.

Loading validates declared structure and file sizes without re-reading source
checkpoints or guessing tensor conventions. It does not hash weight payloads;
download packages from a trusted source. This repository runs prepared packages
and does not include package construction tools.

## Build from source

Source builds require Rust and Apple's developer command-line tools and SDK:

```sh
cargo build --locked --release
```

The executable is `target/release/moespresso`. It can run outside the source
checkout with an explicit package path.

## Correctness and documentation

Changes to quantized math, attention, routing, package loading and cache state
are checked against reference logits and generated-answer tests. Cache tests
exercise conversation continuation through resizing, interleaving, disk restore
and interrupted requests. Performance changes use matched artifacts and explicit
A/B conditions.

- [Validation](docs/testing.md): regular tests, real-package checks, quality
  gates and OpenCode acceptance.
- [Client setup](docs/clients.md): request options, tool calls, streaming and
  cache reporting.
- [Adaptive memory](docs/adaptive_memory.md): admission, expert pools and
  pressure response.
- [Cache-Prior routing](docs/cache_prior.md): resident preference and eviction.
- [Disk checkpoints](docs/disk_checkpoints.md): conversation persistence and
  recovery.
- [Speed measurement](docs/speed_comparison.md): timing and comparison rules.
- [Contributor guide](AGENTS.md) and [distribution](docs/distribution.md):
  development and packaging.

## Acknowledgements

MoEspresso builds on a large body of open work:

- The [Qwen team](https://huggingface.co/Qwen) trained and released the model.
- [Iwan Kawrakow](https://github.com/ikawrakow) designed the IQ_K formats and
  published their reference implementation in
  [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp).
- The [MoEspresso M1 fork](https://github.com/ashe-wb/moespressoV3-m1-fork)
  contributed improvements to routing, kernels and expert loading.
- [Asher Feldman](https://github.com/asher)'s
  [mlx-kquant](https://github.com/asher/mlx-kquant), Apple's
  [MLX](https://github.com/ml-explore/mlx) and
  [MLX LM](https://github.com/ml-explore/mlx-lm),
  [mlx-serve](https://github.com/ddalcu/mlx-serve) and
  [ds4.c](https://github.com/antirez/ds4) provided implementations adapted for
  the native runtime.
- [Georgi Gerganov](https://github.com/ggerganov),
  [ggml](https://github.com/ggml-org/ggml) and
  [llama.cpp](https://github.com/ggml-org/llama.cpp) established the
  block-quantization ecosystem and important reference implementations.
- The [KVarN authors](https://github.com/huawei-csl/KVarN) developed the cache
  codec adapted for native Rust and Metal execution.

## License

MoEspresso is dual-licensed under [Apache 2.0](LICENSE-APACHE-2.0) or
[MIT](LICENSE-MIT), at your option. Third-party code notices are in
[THIRD-PARTY-NOTICES](THIRD-PARTY-NOTICES).
