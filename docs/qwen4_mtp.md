# Experimental full-resident one-draft Qwen4 verification

The Qwen4 full512 IQ_K adapter provides an experimental greedy MTP command for
research, separate from ordinary serving. All of the target's routed experts
must fit in memory; the command refuses bounded residency.

```bash
uv sync
uv run --locked moespresso-qwen4-mtp /path/to/package \
  --sidecar /path/to/mtp-sidecar \
  --prompt "Write a Python function that merges overlapping intervals." \
  --memory-gb 96 --max-tokens 128 --json-out result.json
```

The sidecar is a separate model artifact with a compatible `mtp_manifest.json`,
leaving the target package unchanged. `uv sync` installs the runtime; drafter
weights must be supplied separately. Ordinary serving does not automatically
select this path.

Both verification rows use the model's unbiased one-row router reduction, with
separate small router matmuls to retain ordinary BF16 rounding. Their expert
union and resident slot addresses stay on the device. Shared gate/up weights
are decoded once for both input vectors, while down projections retain
separate reductions. Shared experts, GDN projections, QSA input projections
and the vocabulary head receive two rows so their quantized matrix operations
can reuse weights. Attention and recurrent updates retain the first-row
checkpoint for rejection; second-row attention rollback copies are GPU
dependencies of the append and require no host wait at each attention layer.

Eligible two-row target verification uses one compiled numerical graph. The
graph receives token ids, semantic positions, the physical frontier, PLE rows,
and request-private mixer arrays as explicit inputs, and returns both candidate
state checkpoints. QSA publication remains outside the graph and uses the same
transactional undo, finite-value check, and commit path as ordinary MTP.

The compiled path supports a base frontier of at most 2046 tokens, where QSA
selects the complete visible prefix and KVarN retains the selected keys and
values in its exact BF16 sink and suffix. A fixed 2051-row attention bank lets
frontiers change without retracing the graph. Longer contexts and other QSA
state backends automatically use the existing verifier. Both require full
target expert residency with cache-conditioned routing disabled. The compiled
core is created only when an eligible round is reached;
`MOESPRESSO_QWEN4_MTP_COMPILED=0` selects the existing verifier for every round.

The compiled verifier submits its complete numerical graph while the host
prepares transactional QSA publication. The existing verifier instead submits
completed non-final four-layer prefixes to the MLX stream without waiting,
counting them in the command's `prefix_submissions` output. Both paths complete
candidate logits and state checkpoints before acceptance, and commit and
cleanup retain their request drains.

Acceptance commits only the anchor and accepted draft prefix to target state,
without an expert scratch copy or residency transaction.

The memory planner reserves the resident drafter payload and a conservative
allowance for its attention state before sizing target expert pools, then
requires capacity for all 512 target experts. Context defaults to 4096 tokens
and can be selected with `--max-context-tokens`.

Each run uses fresh request state without publishing disk or prefix-cache
entries, and prints text plus elapsed timing, acceptance and memory totals.
`compiled_verification_rounds` and `fallback_verification_rounds` report which
verifier handled each two-row round. The number of experts shared between rows
is reported as unavailable because collecting it would add diagnostic work to
the measured verification path. The reported measurements describe the run
without guaranteeing higher throughput.

## Bounded ordinary execution

When the ordinary Qwen4 request satisfies the structural lane requirements, the
pipeline keeps one request owner alive while preparing the following row.
Pool publication joins and the dependency on the sampled token remain in place.
