# Qwen4 architecture and serving

`runtime/qwen4` implements the Qwen3.8-Flash-Next text architecture identified
by `qwen4_exp` and `qwen4_exp_text` manifests, separately from the
`runtime/qwen` adapter used by Ornith. The package manifest selects the adapter
regardless of the directory name.

Loading requires a complete MoEspresso package containing its manifest,
tokenizer, model shards and PLE payload; neural weight shards alone are
insufficient. This package/runtime contract does not make arbitrary upstream
checkpoints loadable or declare a new public download.

## State and storage

The full512 graph has 48 layers and selects ten routed experts per layer. Its
hybrid state combines Gated DeltaNet recurrence, sparse QSA attention, and PLE
token/convolution history, all of which must describe the same committed token
frontier.

| Component | Runtime treatment |
| --- | --- |
| Dense projections, routers, norms and shared experts | Resident weights in the package's declared formats |
| Routed experts | Per-layer pools; full residency or demand loading from SSD |
| PLE/n-gram tables | Package-backed selected-row reads; no full GPU residency requirement |
| QSA cache | Mutable KVarN K4/V4 body with exact BF16 sink and recent suffix |
| GDN and PLE state | Included with QSA in composite prompt snapshots |

KVarN preserves an exact 128-token sink and at least the most recent 8,192
tokens, packing older body rows at tile boundaries. Short contexts do not
exercise the packed body. Ornith uses a separate Q8 attention-cache
implementation whose policy does not describe Qwen4.

The capacity planner reserves the configured context's KVarN buffers as well
as non-routed weights, workspace and safety headroom before allocating expert
slots. Increasing context can therefore reduce expert capacity. PLE/n-gram
files remain on SSD and are not charged as fully resident expert weights.
`--max-memory-gb` controls the startup planner without imposing a process-RSS
limit. Retained in-memory prompt snapshots have their own
`--prompt-cache-bytes` cap.

## Ordinary generation

```bash
moespresso generate PACKAGE --prompt "Explain a binary search." \
  --thinking off --max-tokens 128
moespresso serve PACKAGE --thinking off
```

Serving targets 128K context unless an explicit supported limit is supplied.
Thinking defaults to on for this adapter; `--thinking off` selects the
non-thinking template. The packaged generation configuration supplies sampling
defaults unless the request overrides them.

Bounded full512 pools automatically use factor-two cache-conditioned routing
with two protected routes, while full residency remains unbiased and prefill
always uses original routing. See [Cache-Prior routing](cache_prior.md) for the
numerical tradeoff and the `--cache-routing off` override.

Ordinary decoding automatically prepares one row ahead when more than one
output token is requested, no history-dependent logits processor is active,
and the state supports the batch-one, unpadded single-token append lane.
The next row depends on the sampled token and uses no draft-model prediction.
Request ownership and slot publication protect in-flight readers; requests
outside these conditions use serial decoding. Stopping drains and discards
unpublished work.

Full-resident prefill uses packed IQ_K matrix tiles when codec, geometry and
slot-map requirements hold. Smaller calls and unsupported layouts retain
their codec-specific paths. The runtime selects these paths automatically,
without requiring users to enable research flags.

## Prefix reuse and speculation

The HTTP adapter supports in-memory prompt reuse and default-on
[KVarN4 disk checkpoints](disk_kv.md#qwen4-kvarn4-checkpoints). Snapshots contain
the completed prefill state, with packed body data staying packed across save
and restore. On the next turn, generated text is processed as an unbiased
suffix.

Checkpoints omit expert residency and LFU counts, so a restored bounded request
can have different cache-biased decode choices even when its restored prompt
tensors are exact.

MTP is off in ordinary Qwen4 serving at every context length, including
short contexts. The retained [full-resident MTP command](qwen4_mtp.md) is an
explicit experimental path requiring a separate sidecar; its compiled verifier
is available through that command and does not activate MTP in `serve`.
