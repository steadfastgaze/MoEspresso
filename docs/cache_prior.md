# Cache-Prior routing, expert pruning, and prefetching

An MoE model selects a small set of experts for each token, but their weights
may be much larger than available memory. MoEspresso keeps bounded expert pools
in unified memory and loads missing experts from the package on SSD. There are
several ways to reduce those loads: change which experts run, change which
experts remain resident, or load predicted experts early. These are separate
decisions with different effects on model output.

MoEspresso's cache-conditioned routing currently applies to the Qwen4 full512 adapter.
Pool and loader mechanisms
at [SSD streaming](ssd_streaming.md).

## Mixture of Cache (Cache-Prior)

Already the original [Mixture of Cache-Conditional Experts for Efficient Mobile Device Inference,
section 3.3](https://arxiv.org/html/2412.00099v2#S3.SS3) biases expert selection
toward weights already in memory. It adds a cache-dependent bonus to router
logits for ranking, but uses the original router scores to weight the selected
experts' outputs. The bonus is `lambda * average_logit_range`, with the range
estimated per layer across tokens and sequences. The paper also allows the
strongest original routes to be protected by including them in the promotion
mask.

This changes the expert access sequence, so its gains are not limited by the
best possible cache eviction policy for the *original* sequence. It is a
quality/speed tradeoff.

## MoEspresso's implementation

The implementation retains the ranking-versus-contribution distinction, with
a fixed probability multiplier rather than a running logit-range estimator.
For one eligible decode token:

1. Compute original FP32 router probabilities `p = softmax(logits)` and their
   original top ten experts.
2. Reserve the strongest `J` original routes, whether resident or not.
3. For the remaining positions, rank an expert by `f * p[i]` when all three
   projections are resident in the published snapshot, otherwise by `p[i]`.
4. Select ten experts in total. Compute contribution weights from the
   **original** probabilities, normalized over this selected set, then cast
   those weights to BF16.
5. Load any missing selected experts and execute them normally.

Defaults are `f = 2` and `J = 2`. The multiplier is configurable from 1 to 8;
the protected count is configurable from 0 to 3. A protected route is a place
in the current token's selection. Ten experts still execute.

In exact arithmetic, multiplying a probability by `f` for ranking corresponds
to adding `log(f)` to its logit. The kernel deliberately operates on FP32
probabilities and there is no effort of bitwise equivalence after
underflow or ties. There is no running range estimate, learned cache-prior
parameter, or per-request training step. Protection is explicit: protected
routes cannot be displaced by another expert's multiplier.

**Prefill is not biased.** Every prompt chunk follows original routing,
including a one-token chunk. Layers 0 and 1 also retain original routing during
decode. Bias is considered only on eligible single-token decode calls in the
remaining layers. If every unprotected original selection is already
resident, the selector keeps the original selection.

The default `auto` policy enables the 2/2 preference when the resolved Qwen4
expert pools are bounded, while fully resident execution remains unbiased. The
capacity decision includes the configured context's memory reservation.

While a pool is being updated, selection can reuse its previous published
snapshot. The demand loader still resolves every selected expert and waits
for the correct bytes before execution, so a stale hint can cause
a miss but cannot substitute an unrelated expert's weights.

Changing the selected set can change logits and answers even though the
selected experts keep their original contribution scores.

## A dynamic REAP?

Cerebras's [REAP paper](https://arxiv.org/abs/2510.13999) describes one-shot
expert pruning. Its saliency criterion combines router weights with expert
activation norms measured on calibration data. Low-saliency experts are
removed from the model, while the router retains independent control over
the surviving experts.

Cache-conditioned routing can be understood as a **soft, dynamic REAP-like
restriction**: favor a useful working subset instead of continually fetching
every expert the original router would choose.

| Property | REAP pruning | MoEspresso cache-conditioned routing |
| --- | --- | --- |
| Expert availability | Pruned experts are removed | All package experts remain available on SSD |
| Change timescale | A fixed subset for the built model | Selection on each eligible decode token; residency changes with demand |
| Constraint | Cannot use a removed expert | A sufficiently strong nonresident expert can still win |
| Storage effect | Smaller model package | Unchanged package; bounded in-memory working set |

## Comparison with Apple's AFM 3 Core Advanced

Apple describes [AFM 3 Core Advanced](https://machinelearning.apple.com/research/introducing-third-generation-of-apple-foundation-models)
with "Instruction-Following Pruning". A lightweight
dense block selects experts *per prompt, with a periodic reselection during
generation*.

The shared objective is less frequent weight movement.

## Eviction policy

The serving default is **decaying LFU with an LRU tie-break**:

- Demand hits and newly reserved misses update frequency and recency.
- By default, every 128 touches halves the frequency counters.
- When no free slot remains, the lowest-frequency eligible expert is evicted;
  ties choose the least recently used expert.
- Current demand and explicitly protected in-flight readers cannot be evicted.

An explicit LRU alternative exists in the pool API, but it is not the serving
default.

Live demand statistics remain in memory.

## Implementation map

- `runtime/qwen4/cache_routing_config.py`: defaults, parameter bounds and
  routing-policy identity.
- `runtime/qwen4/cache_routing_kernel.py`: protected selection and original
  contribution weights.
- `runtime/qwen4/cache_routing.py`: immutable published residency hints.
- `runtime/qwen4/model.py` and `generation.py`: unbiased prefill and decode
  integration.
- `runtime/expert_slot_pool.py`: LFU decay, victim selection and safe loading.
- `runtime/pooled_switchglu.py`: prefill chunk loading and prefetch tickets.

Paths are relative to `src/moespresso/`. See [cache-routing controls](cache_routing.md)
for commands, policy identity and health counters.
