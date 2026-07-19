# DeepSeek-V4-Flash speed record

DeepSeek speed claims are tied to a package identity, prompt, generation shape,
quality state, runtime mode, memory cap, and host conditions. A primitive or
microbenchmark can justify an experiment; only an engaged served-path run can
justify a product claim.

This document is the public benchmark protocol and the current validated
record. Numeric anchors belong here rather than in subsystem reference docs.

## Package and prompt anchor

The full-resident record uses the byte-faithful package:

- 47 shards, about 80.0 GiB;
- routed expert bytes copied from the matching GGUF;
- optimized DeepSeek kernels expected by the manifest;
- the committed `long_code_audit.txt` prompt;
- 3,844 tokens after one render;
- greedy decode, temperature 0, 64 generated tokens.

Prompt path:

```text
src/moespresso/correctness/fixtures/deepseek_v4/test_vectors/prompts/long_code_audit.txt
```

Package size is not package identity. Record the manifest artifact id and file
hashes for every run.

## Quality anchors

Before retaining a speed change that affects model math or execution order, run
the relevant gates in [`deepseek_v4_quality.md`](deepseek_v4_quality.md).

The current record is associated with:

- Q1 at 17/17 on its certified MLX wheel lattice;
- Q2 NLL anchored per exact MLX wheel;
- Q3 at 16/16;
- token or logit identity checks for route-specific changes.

An isolated free-generation token difference is not enough to diagnose a
quality regression. Low-bit packages sit near decision boundaries; use the
numeric gates and record the MLX wheel.

## Environment discipline

The headline measurements used an M3 Max with 40 GPU cores and 128 GB unified
memory. Absolute runs were taken on AC power, at nominal thermal state, with an
otherwise idle host and an exclusive GPU window, and with the disk KV tier off
(`MOESPRESSO_DISK_KV=off`; it now defaults on when serving).

For comparisons:

- use one fresh process per arm;
- interleave candidate and reference arms in the same session;
- use the same package, rendered prompt, and generation shape;
- capture counters that prove the intended route engaged;
- compare matched pairs rather than numbers from different sessions;
- separate first-token latency, prefill compute, and steady-state decode.

## Full-resident headline

On the 3,844-token anchor, the certified matched pairs were:

| Metric | MoEspresso | DwarfStar reference | Interpretation |
|---|---:|---:|---|
| Decode pair 1 | 26.06 tok/s | 24.77 tok/s | Same host and prompt, one process per arm. |
| Decode pair 2 | 26.24 tok/s | 24.83 tok/s | Repeated certified arm. |
| End-to-end prefill | 14.667 s median | about 13.5 s | MoEspresso within 1.09x. |
| Pure chunk compute | 14.32-14.38 s | reference implied by the same run | MoEspresso within 1.065x. |

At about 8.4 GB of routed weight traffic per generated token, the certified
decode pairs imply about 217 GB/s effective weight bandwidth for MoEspresso and
about 208 GB/s for the reference.

Later standalone sessions measured 25.378-25.452 tok/s, with median 25.436.
That band records environment drift; the matched pairs above remain the public
comparison.

## Prefill and memory

The anchor first-token wall decomposed into about 0.06 s setup, 14.32-14.38 s
chunk compute, and 0.24 s final-token work. The recorded anchor chunk wall was
14.538 s.

Longer prefill points on the banded-offset default were:

| Context depth | First-token wall | Transient MLX peak | Comparison |
|---:|---:|---:|---|
| 7,698 | 34.67 s | 10.74 GiB | 1.10x faster than the composed chunked lattice. |
| 15,406 | 72.22 s | 11.08 GiB | 1.13x faster than the composed chunked lattice. |

Full-resident model state measured 72.56 GiB at load and 84.94 GiB MLX peak
over the anchor request. Model load measured 18.8-20.3 s. Wired-limit prewarm
moved work to load: anchor TTFT was 14.667 s with it and 17.482 s with its kill
switch.

## Certified context envelope

The cumulative agentic road-test used temperature 0, `top_p=1`, up to 700
generated tokens per request, four real tools, and disk KV with stride 4,096.
Context grew in place; the only process restarts were scripted restore checks.

| Metric | Record |
|---|---|
| In-place envelope | 113,855 tokens grown from zero, no aborts and no mitigation restart cadence. |
| Run shape | 61 requests, 1,062.5 s wall, three server segments. |
| Cache accounting | 57/57 in-memory hits at the exact previous full-plus-completion length, two misses, two disk hits, zero ledger mismatches. |
| Frontier storage | 31 checkpoints from 4,096 through 110,592 tokens, 40.0 GB total, zero evictions or quarantines. |
| Deep-turn cost | An 8K suffix with two checkpoint writes took 58-64 s at 81K-106K restored context; first-token latency was 57.9-61.2 s. |
| Live cache | 4.47 GB for the single chain entry at 113,855 tokens. |
| Restart identity | Both scripted restores landed on the predicted 12,288-token frontier; the identical-geometry replay pair was byte-identical. |

The 19-21 tok/s readings inside the agent loop are informational. They did not
use the thermal gate required for a deep-context decode record.

## Bounded expert residency

The 64 GB-class measurements imposed an expert-pool capacity on the 128 GB
host. The package could remain in page cache, so SSD stall seconds are lower
bounds. The table establishes runtime behavior and memory fit. Real 64 GB
storage latency still needs physical-hardware validation.

Medians of three, anchor depth 3,844 and second point at 7,698:

| Expert cap | Anchor TTFT | Anchor decode | Depth TTFT | Depth decode | Anchor MLX peak |
|---:|---:|---:|---:|---:|---:|
| 512, full resident | 14.752 s | 25.42 tok/s | 38.266 s | 24.484 tok/s | 84.94 GiB |
| 56 | 21.726 s | 7.894 tok/s | 60.391 s | 8.665 tok/s | 57.82 GiB |
| 48 | 23.038 s | 6.764 tok/s | 66.117 s | 7.345 tok/s | 49.89 GiB |
| 40 | 24.820 s | 5.687 tok/s | 73.387 s | 6.411 tok/s | 41.95 GiB |

The closing cap-48 state measured 22.832 s anchor TTFT at 7.139 tok/s and
50.17 GiB peak; at depth it measured 63.974 s and 7.999 tok/s. Caps 40, 48,
and 56 reproduced the certified token rails, and cap 48 reproduced the full
quality anchors. A package-vendored hotlist reduced cold first-request expert
misses from 16,588 to 14,044 and raised the request hit rate from 0.540 to
0.595.

Real 64 GB hardware remains the required next validation for stall and latency
claims.

## Run the served snapshot

A bounded structural snapshot:

```bash
uv run --locked moespresso-ds4-speed-stats \
  <package-dir> \
  --prompt-file \
    src/moespresso/correctness/fixtures/deepseek_v4/test_vectors/prompts/long_code_audit.txt \
  --max-tokens 2 \
  --json-out <snapshot.json>
```

The full orientation shape uses `--max-tokens 64`. Add
`--max-memory-gb <budget>` for a bounded-residency run. The command renders a
user prompt exactly once; use `--rendered-prompt-file` only when intentionally
supplying already-rendered input.

## What to record

For every retained change, record:

- package manifest id and package mode;
- MLX wheel and relevant environment controls;
- prompt source, rendered token count, and generated-token cap;
- full-resident or bounded-residency mode and effective expert cap;
- Q1/Q2/Q3 state or the reason a narrower check is sufficient;
- engagement counters proving the candidate route ran;
- first-token wall, prefill compute, decode rate, and memory peak;
- same-session reference arms when making a comparative claim.

If the route did not engage, the result is invalid regardless of speed.

## Known gaps

- There is no thermal-gated decode record at the 113,855-token envelope.
- The 64 GB-class stall record is emulated on a larger-memory host.
- The t/s value obtained by dividing 3,844 tokens by the anchor TTFT is derived;
  the measured quantity of record is seconds.
- Different prefill chunk geometries can choose different deterministic
  low-bit token rails. Disk restore itself was measured logit-identical against
  the matching in-memory geometry.
