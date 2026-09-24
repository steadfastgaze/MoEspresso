# Timing a chat-completions API

`completions-api-timing` measures an existing streaming
`/v1/chat/completions` server using the client clock and local token counts.
It requires no MoEspresso health endpoints or server timing extensions, and
retains API usage and timing fields separately so they can be compared with the
client measurements, including any disagreements.

```bash
uv run --locked moespresso completions-api-timing \
  --url http://127.0.0.1:8080 \
  --model served-model-name \
  --tokenizer /path/to/matching-tokenizer-or-package \
  --output api-timing.json
```

Supply the required tokenizer files matching the served model. An existing
MoEspresso package directory works for comparisons between engines serving
that model, as does an already-cached Hugging Face tokenizer identifier.
Loading is local-only, runs no remote code, and loads no model weights. The
report records the tokenizer class, library versions, vocabulary/template
hashes and the fast tokenizer's complete pipeline hash. `--model` can be omitted
when the server does not require a model name. Both a server origin and its
`/v1` URL are accepted.

After one short arithmetic warmup, the command makes two repeat-major passes
over three standalone default prompts: history, coding and biography. This
protocol differs from `moespresso speed`; each measured request has a
256-token cap. `--repeats 1 --max-tokens 128` shortens the run. Repeat `--prompt`
to replace the default questions; `--label` records the engine configuration. Requests use
temperature zero and top-p one, without top-k/min-p extensions. The command
does not start, stop or reconfigure the server.

## Measurement behavior

Incoming SSE frames receive monotonic timestamps. Prompt counting finishes
before the first request, and output counting runs after each response closes,
before the next request starts. The command makes no health queries, tokenizer
endpoint calls or model-discovery calls, and uses no polling loops or per-chunk
console writes.

Every completed request records:

- The local chat-template prompt count and a hash of its token IDs.
- Independently retokenized content and reasoning counts, their total, and a
  hash of those token IDs. No special tokens are added during this count.
- Response-header time, first nonempty output time, last output time, finish
  event time and full request duration.
- A cumulative local token count at each text-delivery boundary, plus median
  and p95 delivery gaps between chunks. Individual token latencies are not
  measured.
- Local output tokens divided by full request duration.
- A post-first-delivery rate estimate: remaining local tokens divided by the
  time between first and last text delivery. This excludes every token in the
  first batch, which may contain more than one token.
- Local prompt tokens divided by time to first output, labelled as an
  estimate including queueing, transport, cache effects and the first output
  batch; it cannot isolate the prefill-kernel rate.
- API usage, native `timings` fields, and differences between API counts and
  local counts. The API-based request rate is a separate cross-check.

Fast tokenizers map complete-text token offsets back to stream boundaries,
accounting for tokens that span fragments without counting each fragment
separately. Slow tokenizers use post-response prefix retokenization; boundary
counts are estimates because appending text can change a tokenization boundary.

The JSON report includes responses and summary medians excluding warmup. Local
accounting continues even when API counts are missing or invalid. If a server
explicitly rejects `stream_options` during warmup, the command records one
retry without that option and keeps local counting enabled; other failed
requests are not retried. Ctrl-C, truncated streams and errors retain partial
text and measurements with a failed or interrupted status. Existing reports
are never overwritten, and missing API values and unmeasurable rates remain
null rather than being recorded as zero.

## Precision and comparison

Compare engines using the same client metrics: request duration, first visible
output and local returned-text counts are the primary observations.
Post-first-delivery throughput estimates can measure buffering or speculative
token batches instead of internal decode work. A single output delivery cannot
provide that rate; its full-request throughput is still reported. Final usage
or connection-close delays do not extend the first-to-last-text interval.

Retokenizing returned text cannot recover hidden reasoning, stripped control
markers, removed stop tokens, or the exact token IDs the server generated.
Even a matching tokenizer can produce a different segmentation of decoded
text. Inspect differences between API and local counts before attributing them
to a counting bug in the server or client.

Local prompt counts require the server's chat template and options to match.
`--chat-template /path/to/template.jinja` selects a local template override.
`--chat-template-kwargs '{"enable_thinking":false}'` applies those options
locally and sends them to servers supporting that extension. A tokenizer
without a chat template requires the explicit template file; the client does
not substitute an unformatted word or character estimate for prompt tokens.

Keep the model, thinking mode, output limits, cache conditions and sampling
comparable. Record differences in quantization, routing bias, drafting, memory
limits and server options in the label. Disable the disk KV tier on MoEspresso
for speed comparisons unless caching is the subject. No API-only client can
confirm the server's internal configuration. Other traffic, network latency
and buffering can affect the observations. The command measures timing and
does not assess model quality.

For a short stdout speed summary using MoEspresso's own timing fields and one
health preflight, use [`moespresso speed`](diagnostics.md).
