# Client setup

The native server listens on loopback and exposes `/health` and
`/v1/chat/completions`. Start it with a compatible Qwen3.8 package:

```sh
./target/release/moespresso serve /path/to/package
```

The server does not authenticate requests. Keep it on loopback and do not expose
the listener to an untrusted network.

## OpenCode

Add a provider and model entry to the OpenCode JSON configuration. This example
uses the OpenAI-compatible provider, declares a client-side model alias, and
sets thinking options for the model variant:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "enabled_providers": ["native"],
  "model": "native/qwen-local",
  "small_model": "native/qwen-local",
  "provider": {
    "native": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "MoEspresso local",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1",
        "apiKey": "local-placeholder"
      },
      "models": {
        "qwen-local": {
          "name": "Qwen local",
          "tool_call": true,
          "reasoning": true,
          "temperature": true,
          "limit": { "context": 16384, "output": 512 },
          "variants": {
            "medium-thinking": {
              "reasoningEffort": "medium",
              "chat_template_kwargs": { "enable_thinking": true }
            },
            "no-thinking": {
              "reasoningEffort": "medium",
              "chat_template_kwargs": { "enable_thinking": false }
            }
          }
        }
      }
    }
  }
}
```

The context and output limits in OpenCode are client metadata. Set the context
limit to no more than the loaded package and server `--max-context` limits. The
model name `native/qwen-local` selects the configured OpenCode provider/model;
the server uses its single loaded package and accepts the supplied model string
as an alias. The API key is a placeholder required by the generic provider, not
a credential. The example leaves OpenCode permissions to the user's normal
configuration.

MoEspresso does not implement `GET /v1/models`. Configure the model explicitly
instead of relying on server model discovery. The repository's OpenCode
acceptance setup also sets `OPENCODE_DISABLE_MODELS_FETCH=1` to avoid a client
model-list fetch; this is an OpenCode setting, not a MoEspresso option. See
[native validation](testing.md) for the optional OpenCode acceptance command.

OpenCode variant options map to the native request fields. `reasoningEffort`
uses `low`, `medium` or `xhigh`; `chat_template_kwargs.enable_thinking` must be a
boolean. The server defaults thinking on unless started with `--thinking off`,
and a request's `enable_thinking` value overrides that default.

## OpenAI-compatible requests

For a streaming request with cache usage in the final event:

```sh
curl -N -sS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "local",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 128,
    "temperature": 0,
    "stream": true,
    "stream_options": {"include_usage": true},
    "reasoning_effort": "medium",
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

`model` is an alias for the loaded package, not a model lookup. `reasoning_effort`
accepts `low`, `medium` or `xhigh`; `chat_template_kwargs` accepts boolean
`enable_thinking` and `preserve_thinking`, plus `reasoning_effort`. For example,
set `enable_thinking` to `true` to override a server started with
`--thinking off`.

## Status, queueing and cache evidence

Read `/health` for the loaded package artifact ID, capacity, context limit,
prefill width, routing and cache tiers:

```sh
curl -sS http://127.0.0.1:8080/health
```

Health requests use a separate dispatcher and can be served during inference.
Chat completions execute serially. Up to eight further chat requests can wait in
the queue; a request arriving when that queue is full receives HTTP 503.

Completion usage includes `prompt_tokens_details.cached_tokens` and a
`prompt_cache` object. It reports the initial lookup event (`cold_start`,
`memory_hit`, `disk_hit` or `disabled`), restored tokens, committed disk
checkpoint writes, the last checkpoint frontier, checkpoint write time and
whether adaptive serving changed capacity during generation. Streaming clients
receive usage in the final event when `stream_options.include_usage` is true.
The `moespresso` timing object also reports `prefill_tokens` (newly evaluated
prompt tokens) and `prefill_seconds` (completed prompt processing, including
checkpoint writes and fused first sampling, excluding restoration). Exact
cache hits have zero prefill work. `ready_to_first_token_seconds` includes
cache lookup and restoration; `first_token_seconds` excludes them. Neither
includes HTTP queue time or network delay.
`generation_seconds` ends after model generation, before final cache maintenance;
checkpoint write time is reported separately in `prompt_cache`.
Default-on caching does not promise a hit for every prompt. Exact token prefixes
are required, and eviction or size limits can cause misses. Expert-capacity
changes preserve reusable conversation state.
See [disk checkpoints](disk_checkpoints.md) for cache scope and lifecycle.

## Supported protocol subset

The server supports only `GET /health` and `POST /v1/chat/completions`; it does
not provide `/v1/models` or general OpenAI API discovery. Chat requests need a
non-empty message list and text content. Function tools and tool calls are
supported with `tool_choice` set to `auto` or `none`. Only one choice is
supported (`n=1`). Use either `max_tokens` or `max_completion_tokens`, plus the
selected sampling controls `temperature`, `top_p`, `top_k`, `min_p`,
`presence_penalty`, `presence_context_size` and `seed`. `reasoning_effort` and
the documented `chat_template_kwargs` are also supported. JSON and SSE
streaming are available; `stream_options` accepts only `include_usage` and
requires streaming. `frequency_penalty` is accepted only as `0`, and
`repetition_penalty` only as `1`; custom stop sequences and structured response
formats are unsupported. Unknown request fields, unsupported message content
and unsupported option values return a request error. Clients that assume the
complete OpenAI API or send extra provider-specific fields may need a narrower
configuration.
