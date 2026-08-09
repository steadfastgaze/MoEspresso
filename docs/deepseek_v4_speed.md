# DeepSeek-V4-Flash speed record

DeepSeek speed claims are tied to a package identity, prompt, generation shape,
quality state, runtime mode, memory cap, and host conditions. A primitive or
microbenchmark can justify an experiment; only an engaged served-path run can
justify a product claim.

This document is the public benchmark protocol and the current validated
record. Numeric anchors belong here rather than in subsystem reference docs.

## Package and prompt anchor

The record anchor is one prompt at one generation shape:

- the committed `long_code_audit.txt` prompt;
- 3,844 tokens after one render;
- greedy decode, temperature 0, 64 generated tokens.

The shipping-package record below uses that anchor. Sections that measure
something else state their own prompt and generation shape.

Prompt path:

```text
src/moespresso/correctness/fixtures/deepseek_v4/test_vectors/prompts/long_code_audit.txt
```

Package size is not package identity. Record the manifest artifact id and file
hashes for every run.

## Quality anchors

Before retaining a speed change that affects model math or execution order, run
the relevant gates in [`deepseek_v4_quality.md`](deepseek_v4_quality.md). That
document also carries the shipping package's readings across the full ladder in
its release package scorecard.

The gates that qualify a speed change are:

- Q1 greedy selected-token identity, quoted with the MLX wheel lattice the run
  records. The count is wheel-keyed: one package scored against one reference
  can land a step apart on two lattices, separated by a single deterministic
  casing knife edge and reproducible on each. A Q1 count quoted without its
  lattice is not a comparable number. The step total comes from the reference
  fixture, not from a constant.
- Q2 NLL anchored per exact MLX wheel;
- Q3 long-context fact recall;
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

## Shipping package record

The release publishes one DeepSeek package,
`DeepSeek-V4-Flash-0731-2.37bpw-MoEspressoV2`. It stores every routed expert
in the IQ_K formats and decodes them through `mlx-iqk`; the dense side is
`q6_K` and decodes through `mlx-kquant`. Model shards total 84.35 GB
(78.56 GiB), or 90.74 GB (84.51 GiB) with the bundled DSpark drafter. The
whole-model rate is 2.37 bits per weight over the served model's 284.335e9
parameters, with the drafter excluded from both sides; routed experts average
2.2491 bpw as a water-filled mix of IQ2_KS on 89 cells and IQ2_K on 40.

Protocol: ten alternating fresh-process arms on the 3,844-token
`long_code_audit.txt` anchor, greedy decode, the disk KV tier off, a thermal
gate before and after each arm, and an 8-token warmup request ahead of each
measured request. The target used the pooled runtime at capacity 256. All 43
routed layers and all 129 projection pools were fully resident and
identity-mapped, with no request-time expert I/O. Every arm emitted the same
38-token reference rail.

| Arm | Decode | Median | Request peak | Load |
|---|---:|---:|---:|---:|
| Drafter off | 26.473-27.562 tok/s | 26.591 tok/s | 86.75 GiB (93.15 GB) | 23.0-23.2 s |
| Drafter on | 32.479-32.724 tok/s | 32.627 tok/s | 92.63 GiB (99.46 GB) | 29.6-30.2 s |

The arms separate without overlap: every drafter-on arm read above every
drafter-off arm, the smallest gap between them was 4.92 tok/s, and the median
gain was 22.70 percent. Every drafter-on arm ran 13 fixed-three rounds,
proposed 39 tokens, accepted 25, used no plain fallback, and prepared reusable
cache state successfully. The drafter is the bundled DSpark sidecar, which the
runtime engages automatically when the wired budget allows.
`MOESPRESSO_DS4_DRAFTER=off` produces the drafter-off arm;
[`speculative_decoding.md`](speculative_decoding.md) covers the selection
rule.

### Full-resident pooled cross-check

A separate five-pair A/B compared the capacity-256 pooled target with the
dedicated resident IQ_K reference on the same artifact and 38-token rail.
The two arms reported different graph contracts: 43 pooled routed layers with
129 identity pools versus 43 dedicated resident routed layers with no pools.

| Target graph | Decode range | Median decode | Median first token | Median load | Request peak |
|---|---:|---:|---:|---:|---:|
| Dedicated resident reference | 26.529-26.804 tok/s | 26.749 tok/s | 15.417 s | 37.569 s | 86.7497 GiB |
| Pooled, capacity 256 | 26.231-26.481 tok/s | 26.431 tok/s | 15.408 s | 24.454 s | 86.7498 GiB |

All ten arms were token-exact. On this anchor, the pooled median decode rate
was 1.19 percent lower, the first-token medians differed by 0.009 seconds, the
request peaks differed by 0.0001 GiB, and median load time was 34.91 percent
lower. Full residency remains the zero-I/O capacity point of the shared pooled
graph with the measured decode delta stated above.

## Effective weight bandwidth

Decode on a fully resident package is bound by routed weight traffic. Multiply
the routed bytes a generated token reads by the decode rate to get the effective
weight bandwidth the run sustained. The routed byte figure comes from the
package census rather than from the file size, because a token reads only the
experts its router selects.

## Certified context envelope

The cumulative agentic road-test used temperature 0, `top_p=1`, up to 700
generated tokens per request, four real tools, and disk KV with stride 4,096.
Context grew in place; the only process restarts were scripted restore checks.

| Metric | Record |
|---|---|
| In-place envelope | 113,855 tokens grown from zero, no aborts and no mitigation restart cadence. |
| Run shape | 61 requests across three server segments. |
| Cache accounting | 57/57 in-memory hits at the exact previous full-plus-completion length, two misses, two disk hits, zero ledger mismatches. |
| Frontier storage | 31 checkpoints from 4,096 through 110,592 tokens, 40.0 GB total, zero evictions or quarantines. |
| Deep-turn shape | A deep turn at 81K-106K restored context spends nearly all of its wall in first-token latency: an 8K suffix and the two checkpoint writes it triggered, not the generated tokens. |
| Live cache | 4.47 GB for the single chain entry at 113,855 tokens. |
| Restart identity | Both scripted restores landed on the predicted 12,288-token frontier; the identical-geometry replay pair was byte-identical. |

The table is a cache and context result. Cache accounting, frontier storage, and
restart identity are properties of the disk KV tier; how far context grows in
place also depends on the resident footprint of the package serving it. Readings
taken inside an agent loop are not a decode record, because the loop does not
apply the thermal gate a decode record requires.

## Bounded expert residency

A bounded run serves a package under a memory budget smaller than its resident
footprint: the runtime derives an expert-pool capacity from that budget, keeps
the non-routed core resident, and streams the routed experts a request misses.
[`ssd_streaming.md`](ssd_streaming.md) covers the capacity math and the
residency policy; `--max-memory-gb` on the snapshot command below selects a
budget.

Two rules hold for any bounded measurement:

- A bounded arm is valid only if it reproduces the full-resident token rail and
  the quality anchors on the same package. Residency policy decides which expert
  bytes sit in memory, not what the model computes, so a diverging rail is a
  defect rather than a residency effect.
- Capping the pool on a larger-memory host emulates the memory fit, not the
  storage. The package can stay in page cache there, so the stall seconds such a
  run reports are lower bounds and the decode rates that follow from them are
  optimistic. A storage-latency claim needs a host whose memory cannot hold the
  package.

Cold-start seeding belongs in the record. A package-vendored expert hotlist
lowers first-request misses and raises the request hit rate, so a bounded arm
has to say which hotlist tier seeded the pools.

The shipping package was also served with every routed layer capped at 64
experts and seeded from its package hotlist. A capacity-256 pooled arm and the
capacity-64 arm emitted the same 38 token ids and decoded-text digest.
Both were fresh processes with the disk KV tier, drafting, lookahead, and
adaptive growth off, an 8-token warmup, and nominal thermal readings before and
after the measured request.

| Pool capacity | Decode | First token | Load | Request peak | Request-time expert I/O |
|---:|---:|---:|---:|---:|---|
| 256 | 26.341 tok/s | 15.436 s | 24.181 s | 86.7498 GiB | none |
| 64 | 4.784 tok/s | 31.628 s | 7.569 s | 29.3751 GiB | 32,616 loads and evictions; 10,872 bundle-row reads |

The measured capacity-64 request used 21,744 sibling projection cache takes,
matching one bundle-row read for each three-projection expert load. This record
proves the bounded graph, exact token rail, and reduced memory footprint on the
public artifact. It ran on a host that could retain the package in page cache,
so its miss latency and decode rate are not a physical-storage performance
claim.

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
- The shipping package record covers one prompt at one context depth. Deeper
  context points on that package are pending.
- No storage-latency measurement exists on a host whose memory cannot hold the
  shipping package.
- The t/s value obtained by dividing 3,844 tokens by the anchor TTFT is derived;
  the measured quantity of record is seconds.
- Different prefill chunk geometries can choose different deterministic
  low-bit token rails. Disk restore itself was measured logit-identical against
  the matching in-memory geometry.
