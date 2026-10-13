# Cache-Prior routing, expert pruning, and prefetching

An MoE model selects a small set of experts for each token, but their weights
may be much larger than available memory. MoEspresso keeps bounded expert pools
in unified memory and loads missing experts from the package on SSD. There are
several ways to reduce those loads: change which experts run, change which
experts remain resident, or load predicted experts early. These are separate
decisions with different effects on model output.

See [adaptive memory](adaptive_memory.md) for expert capacity and host-pressure
handling. The expert pools hold model weights; conversation state belongs to
the separate [prompt and disk caches](disk_checkpoints.md).

## Mixture of Cache (Cache-Prior)

[Mixture of Cache-Conditional Experts for Efficient Mobile Device Inference,
section 3.3](https://arxiv.org/html/2412.00099v2#S3.SS3) biases expert selection
toward weights already in memory. It adds a cache-dependent bonus to router
logits for ranking, but uses the original router scores to weight the selected
experts' outputs. The bonus is `lambda * average_logit_range`, with the range
estimated per layer across tokens and sequences. The paper also allows the
strongest original routes to be protected by including them in the promotion
mask.

This changes the expert access sequence, so its gains are not limited by the
best possible cache eviction policy for the original sequence. It is a
quality/speed tradeoff.

## MoEspresso's implementation

The implementation uses a fixed probability multiplier for ranking. For one
eligible decode token:

1. Compute original FP32 router probabilities `p = softmax(logits)` and their
   original top ten experts.
2. Reserve the strongest two original routes, whether resident or not.
3. For the remaining positions, rank an expert by `2 * p[i]` when all three
   projections are resident in the published snapshot, otherwise by `p[i]`.
4. Select ten experts in total. Compute contribution weights from the original
   probabilities, normalized over this selected set, then cast those weights
   to BF16.
5. Load any missing selected experts and execute them normally.

The factor of two and two protected routes form the fixed 2/2 policy. A
protected route is a place in the current token's selection. Ten experts still
execute, and protected routes cannot be displaced by another expert's
multiplier.

In exact arithmetic, multiplying a probability by two for ranking corresponds
to adding `log(2)` to its logit. The kernel operates on FP32 probabilities;
underflow and ties can make the two formulations differ. There is no running
range estimate, learned cache-prior parameter or per-request training step.

Prefill is unbiased. Every prompt chunk follows original routing, including a
one-token chunk. Layers 0 and 1 also retain original routing during decode.
Bias is considered only on single-token decode calls in the remaining layers.
If every unprotected original selection is already resident, the selector
keeps the original selection. Fully resident execution therefore stays
unbiased.

The selector uses immutable published residency hints. The demand loader
resolves every selected expert and waits for the correct bytes before
execution. A stale hint can cause a miss, but cannot substitute an unrelated
expert's weights.

Changing the selected set can change logits and answers even though the
selected experts keep their original contribution scores.

## A dynamic REAP?

Cerebras's [REAP paper](https://arxiv.org/abs/2510.13999) describes one-shot
expert pruning. Its saliency criterion combines router weights with expert
activation norms measured on calibration data. Low-saliency experts are
removed from the model, while the router retains independent control over
the surviving experts.

Cache-conditioned routing can be understood as a soft, dynamic REAP-like
restriction: favor a useful working subset instead of continually fetching
every expert the original router would choose.

| Property | REAP pruning | MoEspresso cache-conditioned routing |
| --- | --- | --- |
| Expert availability | Pruned experts are removed | All package experts remain available on SSD |
| Change timescale | A fixed subset for the built model | Selection on each eligible decode token; residency changes with demand |
| Constraint | Cannot use a removed expert | A sufficiently strong nonresident expert can still win |
| Storage effect | Smaller model package | Unchanged package; bounded in-memory working set |

## Comparison with Apple's AFM 3 Core Advanced

Apple describes [AFM 3 Core Advanced](https://machinelearning.apple.com/research/introducing-third-generation-of-apple-foundation-models)
with Instruction-Following Pruning. A lightweight dense block selects experts
per prompt, with periodic reselection during generation.

The shared objective is less frequent weight movement.

## Eviction policy

Expert pools use decaying LFU with an LRU tie-break:

- Demand hits and newly loaded misses update frequency and recency.
- Every 128 touches halves the frequency counters.
- When no free slot remains, the lowest-frequency eligible expert is evicted;
  ties choose the least recently used expert.
- Current demand and protected in-flight readers cannot be evicted.

Live demand statistics remain in memory.

## Controls

`serve` and `generate` use `--cache-routing prefer-resident` by default. To
disable the preference and use original routing throughout:

```sh
./target/release/moespresso serve /path/to/package --cache-routing off
```

Expert capacity and routing are separate controls. Changing capacity alters
how many experts can remain resident; `--cache-routing` selects how residency
affects expert selection. See [speed measurement](speed_comparison.md) for
matched comparisons, and [client setup](clients.md) for reported server
conditions.
