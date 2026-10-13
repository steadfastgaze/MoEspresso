# Speed measurement and comparison

## Measure a running server

```sh
./target/release/moespresso speed
```

The client performs one greedy warmup and three topic-switch/same-topic pairs.
Each result prints before the next request; the summary reports median prefill
and after-first-token decode rates. These are independent prompts, without
conversation continuation. A separate approximately 4K-token prefill follows.
Interactive runs offer an approximately 32K-token case; `y` runs it and `n` or
Enter finishes. Non-interactive runs skip that case. Longer cases use fresh
prompt markers by default, generate at most eight output tokens and report
new and cached counts separately from the short-topic medians.

Defaults are buffered HTTP on `http://127.0.0.1:8080`, 256 output tokens and
a 300-second timeout. `--url`, `--max-tokens`, `--timeout` and `--stream` override
these settings. The optional 32K case defaults to a 30-minute timeout; an
explicit `--timeout` applies to every request. A known insufficient server
context limit skips a longer case. Progress goes to stderr, measurements and
longer-test prompts to stdout.

Prefill divides newly evaluated tokens by processing time, including checkpoint
writes and excluding restoration. A complete cache hit reports prefill skipped.
Decode divides tokens after the first sampled token by the remaining server
generation time. First-token time excludes cache lookup/restore; ready-to-first
time includes it. These server timings exclude HTTP queue and network delay.
Missing timings are reported as unavailable.

The client does not start a model or change server routing, capacity or caches.
Use `serve --no-disk-cache --prompt-cache-gb 0` for uncached measurements.
Ordinary runs use server thinking defaults. Reported conditions sample capacity,
context, routing and cache tiers before and after the short suite, without
measuring system pressure or power use.

## Compare serving configurations

Run the following command against each configuration using the same package,
client build and output-token cap:

```sh
./target/release/moespresso speed --compare-id 1
```

The comparison ID fixes request prompts. Comparison requests use greedy
sampling, thinking on, preserved thinking and medium reasoning effort. Check
the reported prompt-token counts for tokenizer or template differences.
Make the same choice about the optional 32K case in both runs.

Run one model server at a time and wait for it to exit before starting the
next. Keep power mode, OS GPU settings and other workloads unchanged. Record
the reported capacity, prefill width, routing and cache settings alongside
the results. An additional pair can reverse the configuration order with a
new comparison ID.

The comparison decode median requires all six topic requests to be uncached,
reach the requested output cap and finish with `length`. Early stops remain
visible separately. Compare matching topic cases and their actual token
counts; generated answers need not be identical.

Effective prefill divides reported prompt tokens by `first_token_seconds`,
including first-token preparation. It is available only when the server
explicitly reports zero cached tokens and valid timing. Compare it with the
same metric, not with pure prefill timing.

A reused ID on the same server can hit RAM or disk cache. Cached requests are
excluded from fresh comparison rates; the client does not delete checkpoints
or clear server caches. Use a new ID for both members of a replacement pair.
