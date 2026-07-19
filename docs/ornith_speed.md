# Ornith performance

Ornith performance claims are tied to the byte-faithful package, prompt shape,
context depth, KV format, runtime capacity, quality gate, and host conditions.
Resident and streamed results are separate operating points even when they use
the same package.

This document owns the current numeric record. Runtime and optimization docs
should describe mechanisms and link here rather than duplicate these anchors.

## Package and environment

The recorded package is about 19.91 GiB and uses a Q4_K_M-class routed-expert
recipe. Its served graph has 40 layers: 10 full-attention layers with head
dimension 256 and 30 recurrent/linear-attention layers, plus 256 routed experts
with top-8 routing and a shared expert. The product KV default is affine Q8 for
the full-attention layers. The manifest declares a 262,144-token limit.

Measurements used an M3 Max with 40 GPU cores and 128 GB unified memory. Absolute
speed runs were taken on AC power at nominal thermal state without a competing
model process or sustained GPU workload, with the disk KV tier off
(`MOESPRESSO_DISK_KV=off`; it now defaults on when serving). Repeat ranges stayed within 0.66
percent. The matrix used three fresh processes per engine and context, with the
engine order rotated between rounds. Each engine received the same prompt token
IDs.

The measured stacks were MoEspresso 1.0.0 with its Q4_K_M package, mlx-lm 0.31.3
with MLX 0.31.2 and Jundot oQ4e, and llama.cpp at `6eddde0` with the Q4_K_M
GGUF. The table compares complete products, so weight artifacts and Q8 cache
codecs are recorded for each arm.

The 32 GB-class result is a capacity-capped run on that 128 GB host. It proves
fit and bounded-residency behavior, while SSD stall time remains a lower bound
because the whole package can stay in page cache.

## Quality requirement

The recorded resident package and the cap-192 streamed operating point both
pass the full gate in [`ornith_quality.md`](ornith_quality.md) at 9/9. Speed
changes that touch model math, rendering, caches, or execution order must clear
the same gate before promotion.

Matched token rails or an explicit gate are required for every A/B. A speed
number without proof that the intended route engaged is not evidence.

## Q8 comparison matrix

Every cell is the median of three fresh-process runs with an eight-token prewarm
and 256 measured output tokens. Values are **decode throughput / TTFT-derived
prompt throughput**, in tokens per second.

| Context | MoEspresso Q8 KV | mlx-lm Q8 KV | llama.cpp Q8 KV |
|---:|---:|---:|---:|
| 3,969 | 84.81 / 1,149.51 | 67.39 / 1,578.30 | 62.36 / 1,129.05 |
| 8,191 | 83.78 / 1,110.66 | 63.33 / 1,496.54 | 58.74 / 1,058.73 |
| 37,000 | 67.31 / 820.00 | 48.09 / 1,101.59 | 43.52 / 693.02 |

MoEspresso records the highest Q8-KV decode throughput at all three contexts.
MoEspresso and mlx-lm use affine Q8 with group size 64. llama.cpp uses `q8_0`
K and V with group size 32. The Q8 policy applies to the ten full-attention
caches; the thirty recurrent state caches retain their normal representation.
MoEspresso and llama.cpp use Q4_K_M weights from the same GGUF lineage. mlx-lm
uses Jundot oQ4e weights.

A separate matched mlx-lm diagnostic at 8,191 tokens measured 69.96 tok/s with
raw BF16 KV and 63.33 tok/s with affine Q8 KV. Raw BF16 uses substantially more
attention-cache storage, so Q8 remains the product comparison. The Q8 arm
selected EOG after a short continuation, while the raw-BF16 arm selected a
different first token. The mlx-lm Q8 result is therefore timing evidence, not a
cache-quality equivalence claim. Cacheless teacher-forced NLL does not measure a
generation cache codec.

## Certified context envelope

The cumulative agentic context test used temperature 0, `top_p=1`, disk KV with
stride 4,096 and a 450 GB byte budget, and no mitigation restarts. Context grew
in place from zero.

| Metric | Record |
|---|---|
| In-place envelope | 160,965 context tokens, no aborts or unscripted restarts. |
| Run shape | 59 requests, 705.7 s wall, three server segments with two scripted restore checks. |
| Cache accounting | 54/54 in-memory hits exact, three misses, two disk hits, zero mismatches. |
| Frontier storage | 46 entries and 39.6 GB total; the 4,096-token entry was 112 MB, including about 67 MB fixed recurrent-state payload. |
| Deep append cost | 10K-token appends at 100K-150K restored context took 30-46 s wall; first-token latency grew smoothly from 32.6 s to 45.6 s. |
| Deep in-memory hit | 0.51 s first-token latency at 161K. |
| Restart behavior | Fresh-process restored first tokens in 25.5 s and 30.2 s; the identical-geometry replay pair was byte-identical. |

Decode inside the agent loop ranged from 60-70 tok/s below 2K context to
20.2 tok/s on the 161K wrap turn. These are informational readings, not
thermal-gated absolute benchmarks.

## 32 GB-class operating point

The cap-192 streamed configuration measured a 19.36 GiB MLX peak at the 37K
point. It uses the same package and bounds runtime expert slots.

| Metric | Record |
|---|---|
| 37K prefill | 560.8 t/s, 66.0 s first-token wall. |
| 4K prefill | 677.4 t/s. |
| Miss behavior | 0.62 hit rate at 37K. |
| Stress validation | Cap 128 remained token-identical at a 6 percent hit rate and 366K misses. |
| Quality | Full gate 9/9 at cap 192. |

Physical 32 GB hardware is required before publishing device-level SSD latency.

## Memory per token

The cumulative context test's live cache grew by about 11.3 KB per token over
its final 61K tokens. At 160,900 tokens the store measured 2.40 GB. At 18,387
tokens, growth from zero measured 14.7 KB per token. Fixed recurrent state is
amortized over a longer sequence.

Raw BF16 attention KV requires about 1.9 times the storage of affine Q8 with
group size 64. This ratio covers the ten full-attention caches only. The thirty
recurrent state caches and all model or working-memory allocations are outside
that comparison.

The serve process RSS stayed in a 9.2-11.3 GB band without a depth trend, but
Metal unified-memory allocations are not fully visible in RSS. The cache-store
series and MLX peak are the useful memory measures.

At 37K the cross-engine adapter measured a 31.15 GiB MLX peak and 20.40 GiB
active after generation in every MoEspresso repeat. The served-path acquisition
boundary measured 23.93 GiB peak at full capacity and 19.36 GiB at cap 192.
These peak values use different acquisition boundaries and are not directly
interchangeable. Chunk size 4,096 is the served operating point.

## Benchmark protocol

There is no dedicated cross-engine Ornith speed CLI. The comparison matrix
used external measurement adapters kept with the raw record. MoEspresso loaded
the verified package through its manifest-driven runtime and exposed exact
numeric token IDs, token-ready timestamps, cache offsets, and engagement
counters. mlx-lm used an equivalent in-process numeric-token adapter. llama.cpp
used `llama-server` with one slot. The full acquisition contract is in
[`benchmark_reproduction.md`](benchmark_reproduction.md).

Operational serving and bounded-residency checks still use the public server:

```bash
uv run --locked moespresso-serve \
  <package-dir> --thinking off
```

For a bounded-residency arm add `--max-memory-gb <budget>`. Use a fresh process
for each arm, wait for startup warmup to finish, preserve the same sampler and
token cap, and record the response usage and timing fields plus server runtime
counters.

For every retained result record:

- package manifest id, file hashes, and MLX wheel;
- prompt identity, rendered token count, sampler, and completion cap;
- full-cap or bounded capacity and measured MLX peak;
- gate result and token-identity evidence;
- first-token wall, prefill throughput, decode rate, and engagement counters;
- matched reference arms when making a comparative claim.

## Known gaps

- The 32 GB-class record is an emulated capacity cap on a 128 GB host.
- The 160,965-token cumulative context test does not provide a thermal-gated
  deep-context decode curve.
- The package is SSD-streaming ready. Rebuilding it directly from the published
  checkpoint requires the source adapter described in
  [`ornith_package.md`](ornith_package.md).
