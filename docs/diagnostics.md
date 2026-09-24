# Checking decode speed

Run `speed` against an already-running MoEspresso server:

```bash
uv run --locked moespresso speed
```

The default address is `http://127.0.0.1:8080`. For a server on another host:

```bash
uv run --locked moespresso speed --url http://server-address:8080
```

The command makes one `/health` preflight request, then prints one summary to
stdout, for example:

```text
MoEspresso speed
Apple M1 Max · 32 GPU cores · 32 GiB
Qwen3.8-Flash-Next · context 131072 · thinking medium
SSD experts 223/512 per layer · auto memory 23.96 GiB · Cache-Prior 2/2
Decode: topic switch 12.31 · same topic 14.18 · overall 13.27 tok/s
TTFT: 0.65 s median · greedy · 6/6 requests · 256-token cap
Conditions: normal
```

The preflight records hardware, model family, context limit, thinking mode,
expert-pool capacity, resolved memory limit, and Cache-Prior policy when
available, without retaining raw system information. If the server does not
expose compatible health data, the speed check continues with conditions
marked as unavailable.

Progress, warnings and errors go to stderr; the command writes no files and
retains no responses. It uses the server's token counts and timing fields, so
the client needs no tokenizer or model weights and can measure any MoEspresso
model architecture that exposes those fields.

The fixed default run sends one unrelated 64-token warmup, followed by three
adjacent pairs of distinct prompts: history, coding and biography. In each pair,
the first prompt measures a topic switch and the second asks about the same
topic. The six measured requests have a 256-token cap, and the summary excludes
warmup. `--max-tokens` changes the measured-request cap, `--timeout` sets each
HTTP request's timeout in seconds, and `--stream` requests SSE in place of the
default buffered responses.

Requests use temperature zero, top-p one, and zero top-k, min-p and presence
penalty. The server's thinking and template settings remain in effect.

Decode speed is calculated for each request as:

```text
(completion_tokens - 1) / (generation_seconds - first_token_seconds)
```

The output reports median decode rates for usable topic-switch requests,
same-topic requests and all measured requests. Server TTFT is the median
server-reported time to its first generated token. These generation timings
exclude HTTP transport time, though network backpressure can affect generation
in `--stream` mode.

A request that produces fewer than two tokens or omits usable timing fields
cannot supply a decode rate, and the summary shows how many measured requests
contributed. After an error or interruption, completed measurements remain on
stdout and the command returns a nonzero status. A run with no usable decode
rate also returns a nonzero status.

Run the server from the version being measured and use matching model, context,
memory, thinking and cache settings for comparisons. The client does not change
server settings, stop other workloads, or clear caches; existing prefix-cache
reuse can affect TTFT.

When the server reports that live available memory limited its expert pool
materially below the normal planner ceiling, the speed check marks the result as
memory constrained and prints a warning. Close memory-heavy applications such
as web browsers, Docker or virtual machines, restart MoEspresso, and rerun
before sharing the result. The warning does not identify a cause beyond the
reduced available-memory condition.

For comparisons with other engines using independent local token counts and
client-observed streaming timings, use
[`moespresso completions-api-timing`](completions_api_timing.md).
