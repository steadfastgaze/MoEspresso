# Runtime: the non-streaming serve path

How MoEspresso turns a `mjtq` package on disk into served tokens, for the
resident (non-streaming) path in `src/moespresso/runtime/`. The artifact-centered
design comes down to one thing here: the engine builds the model from the
manifest. It never sniffs marker files, never infers bit-widths from array
shapes, never reads the source checkpoint. Model-specific execution code is fine;
what is ruled out is model-specific detection and conversion at load.

> Scope note. SSD streaming is the primary MoE runtime; the fully-dense
> package is the explicit non-streaming exception (no routed experts to stream).
> `build_manifest_runtime` (`serve.py`) dispatches: a non-dense package whose
> `required_ops` include `tq_dequant` goes to the streaming builder; everything
> else builds resident via `runtime/build.build_model`. This document covers the
> resident path. Where serve code reaches into streaming state (the truth line,
> hotlist persistence), it is called out but not detailed.

---

## 1. Serving a package: build from the manifest, then generate

A served model is a `(model, tokenizer, manifest)` triple. `serve.load_served_model`
produces it:

1. Read `package_manifest.json` (content-hash-verified by `read_artifact`). A
   directory with no manifest raises `PackageNotFoundError` with a clear,
   traceback-free message (and a hint that packages usually live under the HF
   hub cache when the path was relative).
2. Build the model straight from the manifest via the injectable
   `build_fn(manifest, package_dir)` seam. The default is
   `_manifest_driven_backend`, which calls `build_manifest_runtime`.
3. Print one honest **runtime truth line** stating which runtime the user
   actually got (`_runtime_truth_line`) (`runtime=resident` for this path)
   so nobody has to guess whether they got the resident or streaming engine.

### The build (`runtime/build.py`): the proven jang loader, no dequant at load

`build_model` reuses the serve path already proven on Qwen3.5/3.6 mixed
affine+TQ, driven entirely by the manifest:

- `_runtime_adapter_kind(manifest)` chooses the loader from the **declared**
  `architecture.family` + `required_ops`, never from guesswork:
  - `mjtq_dsv4`: `deepseek_v4_flash`, whose ops must stay within the declared
    DS4 set.
  - `regular_jang_v2`: `qwen3_5_dense` whose ops are a subset of the dense
    affine set (`affine_dequant`, MX-float dequants, passthroughs) → jang's
    `load_jang_model`.
  - `qwen_kquant_moe`: `qwen3_5_moe` with `kquant_dequant` experts (and no
    `tq_dequant`).
  - `jangtq_moe`: any other non-dense family with `tq_dequant` → jang's
    `load_jangtq_model`. (Unknown combinations raise `UnsupportedRuntimeAdapter`.)
- The package carries jang-compatible **sidecars** (`config.json`,
  `jang_config.json`) that convert *generated from the manifest*: a compat view
  for the loader, with the manifest staying the source of truth. The loader
  builds the graph from `config.json` (affine
  non-experts as MLX `QuantizedLinear`; TQ experts as TurboQuant metal-kernel
  modules), and a per-tensor `tensor_map` override (`_apply_tensor_map`) pins
  each affine module's exact `bits`/`group_size` so shape-guessing can never
  pick the wrong precision.
- **No dequant at load.** TQ weights stay packed and the GPU kernel runs them;
  affine modules dequant in-layer at inference. No numpy on this path.
- **Bundle packages.** When the package carries per-layer routed-expert
  *bundles* (no pre-stacked keys), jang's loader leaves a random-init `SwitchGLU`
  in place, *silently*. `_install_routed_experts_from_bundles` owns that
  payload: it reads each layer's bundle once, splits components per the index
  geometry, and replaces each projection with a `TurboQuantSwitchLinear`
  carrying the exact packed/norms bytes (filled by byte-copy into persistent MLX
  buffers, no numpy on the engine path). Anything missing raises
  `RoutedExpertInstallError` rather than serving a quietly-wrong model.
- **Mixed gate/up bits** (`_wrap_mixed_bit_switchglus` + `owned_switchglu.py`):
  jang monkeypatches `SwitchGLU.__call__` at the class level with a fused
  gate+up kernel that has *one* bit-width parameter. For layers whose routed
  gate and up projections deliberately use *different* bit-widths (detected from
  the expert-index headers, `_mixed_gate_up_layers`), that fused kernel is
  invalid. `OwnedSwitchGLU` is a distinct class that keeps the same module
  attributes but owns its own forward (gather-sort → per-projection apply →
  scatter-unsort), immune to the class patch. Layers the metadata declares as
  mixed but that don't wrap raise `MixedBitSwitchGLUError`, fail-loud both ways.
- jang's loader prints a verbose multi-line banner to stdout;
  `_load_jangtq_quietly` captures it and drops it on success, but re-emits it on
  failure so a broken load stays diagnosable.

`build_model` returns `(model, tokenizer)`. The tokenizer is the one jang's
loader produced (mlx_lm `load_tokenizer` + eos/chat handling). The runtime does
**not** re-load it separately; that would diverge from the proven path. (One
upstream false-positive "fix_mistral_regex" warning that transformers emits for
non-Mistral tokenizers loaded from a mixed dir is filtered out by
`_silence_known_transformers_warnings`; the warning is dropped, never acted on.)

### Generation (`serve.generate_with_metadata`)

Generation runs over an **already-rendered** prompt (a string or token ids). It
never templates (see §4). It drives mlx_lm's `stream_generate` with an injected
sampler, accumulating text and token ids, timing **first-token latency** and
total generation time, and returns a `GenerationResult` (`generation.py`): text,
`finish_reason`, prompt/completion/cached token counts, the generated token ids,
the (mutated) `prompt_cache`, and the latency fields. The `stream_generate`/
`sampler` functions are injectable so the contract is testable without MLX.

`generate_once` is the thin string-in/string-out wrapper the CLI uses;
`PrefixCacheGenerator.__call__` is the server's path (§6). Both feed the same
`generate_with_metadata` seam. KV policy (§5) is validated and translated to
mlx_lm kwargs here.

---

## 2. CLI entry points

The four core console scripts, out of the full set declared in
`pyproject.toml` (the model-specific builders, quality gates, and speed probes
are listed in `DEVGUIDE.md`):

| Command | Module entry | Job |
|---|---|---|
| `moespresso-convert` | `package.convert:main` | End-to-end conversion (inventory → probe → optimize → package). |
| `moespresso-generate` | `runtime.serve:main` | One-shot: load a package, render once, generate, print. |
| `moespresso-serve` | `runtime.http:main` | OpenAI-compatible HTTP server over the same load+generate seam. |
| `moespresso-verify` | `runtime.serve:verify_main` | On-demand manifest, member-identity, tensor-key, and sidecar gate. |

### `moespresso-convert`: end-to-end conversion

`package.convert` is the imperative shell that streams the whole pipeline
(`inventory → probe → optimize → package`) on a few-GB machine and writes a
package (quantized shards + `package_manifest.json`) to disk, e.g. an SSD. Every
phase streams (the probe samples by byte-range; the writer quantizes a row-band /
one expert at a time), so a 35B converts in a bounded footprint. It is
**format-neutral**: it consults the *target format's declared* requirements
rather than hardcoding policy: `mjtq` declares it requires `calibration`, so the
CLI always produces a calibrated package; producing an uncalibrated `mjtq` is a
deliberate in-process library call (`allow_uniform=True`), never a CLI accident.
This is the producer of the packages the rest of this document consumes.

### `moespresso-generate`: one-shot

`serve.main`: `moespresso-generate <package_dir> [--prompt ... --max-tokens ...
--temperature ... --top-p ... --thinking off|on|high|max --max-memory-gb ...]`.
It loads via `load_served_model` (builds from the manifest, **does not
verify**), renders the prompt **once** via `http.render_prompt` (the same
render seam the server uses), then calls `generate_once` on the rendered
string. `--thinking` resolves through the family's own mechanism or refuses
loudly (§4); `high` is an alias of `on` for every family. For DeepSeek-V4 the
selection maps onto the official encoder modes: `off` renders chat mode (the
default), `on` renders thinking mode, and `max` adds the official maximum
reasoning-effort preamble. `max` refuses loudly for families without an effort
mechanism.

### `moespresso-serve`: OpenAI-compatible HTTP

`http.main` → `http.serve`: load the package **once** at startup (no verify),
prime one isolated deterministic four-token generation, then serve. The prime
moves first-use model wiring and MLX graph setup before readiness. It bypasses
the prompt-cache manager, publishes no memory or disk KV entry, and does not
persist its synthetic expert demand; in-memory expert residency and runtime
counters may still reflect it. `--startup-warmup off` restores lazy
first-request setup for cold-start measurements. The server prints an explicit
not-ready warmup line and announces readiness only after the prime finishes.
It then binds an OpenAI chat-completions endpoint over the same load+generate
seam (§3). `--thinking off|on|high|max` is resolved to the family's mechanism
**before** the warmup and socket bind (§4). Cache sizing is via
`--prompt-cache-size` / `--prompt-cache-bytes` (both in-memory; §6); these are
host resource bounds and apply to every family. For DeepSeek-V4 the selection
maps onto the official encoder modes at startup and stays fixed for the
server's lifetime; per-request render fields remain rejected, so the DS4
cache/attention policy cannot change between requests.

### `moespresso-verify`: the on-demand integrity gate

`serve.verify_main`: `moespresso-verify <package_dir>`. Reads the manifest, runs
the package and generated-sidecar checks, prints each issue, and exits
**0 = clean / 2 = failed**. This is the gate the serve hot path deliberately
skips (§3).

---

## 3. Verification: fail-closed, off the hot path

`verify.py` is the integrity gate, and it is pure (no mlx/jang), so it is
importable and testable anywhere. `verify_package(manifest, package_dir)` checks,
fail-closed:

1. **Manifest**: base artifact contract, canonical content id, valid status,
   supported package-format version, and no embedded blocking validation.
2. **Files**: every declared shard, tokenizer file, and agentic profile is
   present, package-relative, and matches `size_bytes` and `sha256`.
3. **Tensors**: every declared tensor's expanded on-disk keys (per format:
   `tq` → `tq_bundle`; `affine` → `weight`/`scales`/`biases`; `fp16` → the
   prefix itself) present in its shard's safetensors header.
4. **Generated sidecars**: `config.json` and `jang_config.json` match the
   manifest-derived runtime views semantically.

The checks return `Validation` entries (empty means clean); any blocking entry
makes `moespresso-verify` exit 2. Run it before loading an unverified package.

The full sha256 gate is kept off the serve hot path. Hashing the whole package
(tens of GB) before the first token would dominate startup. So:

- `load_served_model` / `serve()` / `http.serve()` build straight from the
  manifest and **do not verify** on load. (The manifest itself is still
  content-hash-verified by `read_artifact`; it is the *shard sha256s* that are
  skipped.)
- Integrity is a separate package-acquisition gate: run `moespresso-verify`
  after building, downloading, copying, or moving a package; do not pay for the
  same whole-package hash pass on every cold start.

Verification and a fast hot path coexist because they are separate gates. Verify
checks the package once; from then on the engine trusts the declared contract.

---

## 4. The single-render chat contract

**Invariant: exactly one chat-template render per request.** The rendered token
stream is the cache key for KV reuse (§6), so applying the template twice
(`template(template(messages))`) corrupts the token stream and invalidates any KV
prefix keyed on it. There is exactly one render site:

- `http.render_prompt(messages, tokenizer, template_kwargs=...)` is the **single
  place** templating happens. It calls `tokenizer.apply_chat_template(...,
  tokenize=False, add_generation_prompt=True)` once. (A plain role-tagged
  fallback keeps the pure core usable in tests without a tokenizer.)
- `serve.generate_once` / `generate_with_metadata` operate on the
  already-rendered prompt and **must not** call `apply_chat_template`. mlx_lm's
  generate only `.encode()`s a string prompt; it does not re-template.
- The CLI (`serve.main`) renders once via the same `render_prompt`, then
  generates from the rendered string, the identical seam the server uses.

### The vendored per-family chat template

`package/templates.py` + `package/tokenizer.py`: some families need a chat template *MoEspresso*
controls rather than whatever shipped in the source. For the Qwen 3.5/3.6
families (`qwen3_5_dense`, `qwen3_5_moe`) MoEspresso vendors the
community-validated froggeric template (Apache-2.0), which:

- renders history **append-only** (preserves past `<think>` blocks by default),
  so a KV prefix cache stays valid across chat turns;
- keeps the append-only property in thinking-off sessions too: a thinking-off
  generation opens with an empty think scaffold, so replaying that assistant
  turn re-emits the same scaffold. Turn N+1's rendered prompt is then a byte
  prefix extension of turn N's, and the session keeps its cache hit;
- fixes the official template's double-render / agentic-stall bugs;
- is minijinja / C++-safe.

`chat_template_for(family)` returns the vendored text (or `None` when MoEspresso
does not override that family, its source template is then kept as-is).
**Convert** installs it: `copy_tokenizer_into_package` copies the source
tokenizer files into the package and, before hashing, overwrites *both* the
standalone `chat_template.jinja` and the embedded
`tokenizer_config.json["chat_template"]` so the package has one coherent answer
(no stale shadow copy). The manifest records the tokenizer identity (file
hashes + a `rendering_id` covering the installed template + `chat_template_source`).
**Serve** loads the tokenizer from the package, never from the source checkpoint.

### Thinking on/off: per family, refuse loudly

There is no cross-model standard for disabling chain-of-thought, so
`thinking.resolve_thinking_kwargs(tokenizer, thinking=..., family=...)` resolves
`--thinking off|on|high|max` to the **family's own mechanism** (`high` is an
alias of `on` for every family; `max` is DeepSeek-V4's reasoning-effort level
and refuses loudly elsewhere), in order:

1. **Template sniff**: if the package's chat template declares `enable_thinking`
   (Qwen convention), pass the bool through. Zero per-model code; the model
   authors own the semantics. This is the closest thing to a standard
   (vLLM/SGLang expose generic `chat_template_kwargs` for the same reason).
2. **Family adapter table** keyed on the manifest family: one tiny entry per
   ported family that needs something else. DeepSeek-V4 does not use this
   table: the serve layer maps the selection onto its official encoder modes
   directly (`http.deepseek_v4_contract_template_kwargs`).
3. **Refuse loudly**: `ThinkingToggleUnsupported`. A user who asked for
   `--thinking off` must never silently get thinking-on.

Crucially, the server resolves this **at startup, before binding the socket**
(`http.serve`): an unsupported toggle refuses immediately rather than serving the
wrong default on every request. The default (no `--thinking` flag) is the
template's own default. MoEspresso's baseline render kwargs
(`DEFAULT_TEMPLATE_KWARGS = {"enable_thinking": True, "preserve_thinking": True}`)
keep the model thinking-on and history append-only. Generic templates use
`chat_template_kwargs` with precedence
**module defaults < server launch flags < per-request kwargs**. DeepSeek-V4 is
stricter: callers may select `enable_thinking`/`reasoning_effort`, but
`preserve_thinking` and `drop_thinking` are runtime-owned contract fields.
Requests that try to set them fail closed instead of producing a cache-invalid
attention mode.

---

## 5. KV policy (in-memory live KV)

`kv_policy.py` is pure and import-light: it only parses and validates the live-KV
policy MoEspresso owns; actual cache objects and MLX calls live at the runtime
edge. The `KVPolicy` dataclass:

- **`live_kv_format`**: `mlx_affine_q8` (default, symmetric q8) or `raw`
  (explicit fallback). q6 and TurboQuant/vMLX KV are **refused**
  (`KVPolicyError`).
- **`kv_group_size`**: for q8 must be one of `{32, 64, 128}` (default 64),
  enforced fail-closed by `validate_runtime_policy`.
- **`quantized_kv_start`**, **`prompt_cache_size`**, **`prompt_cache_bytes`**.

`parse_kv_policy(request)` reads these names off the request body (so a client
can tune live KV per request), `validate_runtime_policy` fails closed on
unsupported combinations, and `stream_generate_kv_kwargs` translates the policy
into mlx_lm `stream_generate` kwargs (`kv_bits=8`, `kv_group_size`,
`quantized_kv_start` for q8; nothing for raw). `suffix_token_slice` returns the
cache suffix **by token offset**, never by string slicing / re-tokenizing.

This is a **live, in-memory KV quantization policy** for the running cache. The
policy object is not a stored artifact; it is resolved per request and never
serialized.

---

## 6. In-memory prefix reuse

`prefix_cache.py` owns MoEspresso's prompt-cache glue. The store is
**`PromptCacheStore`**: an in-memory, bounded (`max_size` entries /
`max_bytes`) store of prompt-cache objects keyed by token prefix, holding one
live timeline per session chain. A fetch moves the matched entry out of the
store and returns the stored object itself (no deep copy); the generated-
through cache is published back by the insert, after generation completes,
under the serve lock. An insert pops strict-prefix entries of its key
regardless of the caches' trimmability, so an append-only session retains
exactly one entry instead of one snapshot per request (rotating-window caches
report untrimmable, which otherwise measured 3.30 GB retained at 89.8k tokens
under the ten-entry cap). The cost is that a branch from an earlier prefix
loses its in-memory hit and is served by the disk frontier restore, which is
exact. This prefix reuse is **in-memory first**. The disk KV cache adds a
second tier that restores an exact token prefix from disk on an in-memory
miss; serving enables it by default under a per-package root
(`MOESPRESSO_DISK_KV=off` disables it) and it is documented separately in
`docs/disk_kv.md`.

`PrefixCacheGenerator.__call__` per request:

1. **Encode** the rendered prompt to token ids with the same BOS/special-token
   rule MLX uses (`encode_rendered_prompt`), so cache keys match the generation
   token stream exactly.
2. **Refuse an over-limit request** before any cache access. The generator uses
   the effective served context limit: 128K or the package's architecture
   limit, whichever is smaller, unless `--max-context-tokens` explicitly
   selects another positive value up to the architecture limit. It raises
   `ContextLimitError` when prompt tokens plus the requested `max_tokens`
   exceed that limit; the HTTP layer maps the error to a 400. The check runs
   before `fetch_nearest_cache` because the store hands entries out by move: a
   refusal after the fetch would cost the session its chain entry. A package
   without a declared architecture limit uses the 128K default.
3. **Bucket** under `cache_model_key` = `(artifact_id, effective_rendering_id,
   live_kv_format, kv_group_size, quantized_kv_start)`. Token ids alone aren't
   enough: the same tokens under another package, render policy, or KV format
   must never reuse a cache object. The `effective_rendering_id`
   (`http.rendering_identity`) folds the convert-time `rendering_id` (tokenizer +
   installed template hash) together with the effective template kwargs, so a
   cache built under one render policy is never reused under another. Sampling
   knobs deliberately stay out of this key: they are generation-only, so a
   client may vary them turn over turn on one session without losing its
   prefix.
4. **Fetch nearest** prefix from the store → `(prompt_cache, suffix_tokens)`;
   `cached_tokens = full − suffix`. A hit generates only the suffix over the
   reused KV (the append-only template makes follow-up turns extend the prefix).
   Empty-suffix exact hits and misses fall back to a fresh `make_prompt_cache`
   (`exact_fallback`/`miss`), since MLX needs at least one prompt token.
5. **Generate** through `generate_with_metadata` (which MLX mutates the cache
   object in place), then **insert** the cache back under the full token sequence
   (prompt + generated) so the next turn can reuse it.

A client may send `metadata.moespresso_cache_key` on a request to group its
requests as one session chain for disk-cache eviction preference. It is an
index hint only: it is stored on the disk checkpoint entry and never enters
the safety key, so it can never authorize a load, and a request without it
behaves exactly as before.

`cache_stats()` exposes a small snapshot (default/supported live-KV formats,
entry count, byte count) for `/health`, plus a `disk` sub-block when the
disk KV cache is enabled (`docs/disk_kv.md`). Prefix reuse is in-memory
first; the disk KV cache is the one durable prompt-cache tier, on by
default when serving (`MOESPRESSO_DISK_KV=off` disables it). The only
durable artifact otherwise is the package itself.

---

## 7. HTTP layer (`http.py`): thin, pure-core/IO-edge split

The OpenAI-compatible server is deliberately split so the protocol logic is
testable without a socket, MLX, or jang:

- **Pure core**, `chat_completion(request, generate, ...)`: validate the body
  (non-empty `messages`, each with `content`; malformed → `RequestError(400)`),
  parse+validate KV policy and sampling fields, compute the
  `effective_rendering_id`, **render once** (§4), call the injected
  `generate(prompt, ...) -> str | GenerationResult`, and shape an OpenAI
  `chat.completion` dict. A `ContextLimitError` from the generator (§6) maps to
  a 400 client error. Usage includes `prompt_tokens_details.cached_tokens`, a
  `prompt_cache` block (the cache event, entry/byte counts, and, when disk KV
  is on, the `disk_hit` event and a `disk_checkpoints_written` count), and a
  `moespresso` block with first-token latency / generation seconds / tokens-per-
  second (the headline serve metric surfaced on every response).
  The `created` timestamp and the `generate` callable are injected: no
  wall-clock read, no model dependency in the core.
- **Sampling pass-through.** Beyond `temperature`/`top_p`/`max_tokens`, a
  request may set `top_k`, `min_p`, and `presence_penalty`; each is validated
  and forwarded to generation. Fields absent from the request are not
  forwarded, which keeps a request without them byte-identical to the
  pre-existing call shape. Sampling parameters never enter any cache identity.
  `repetition_penalty` has no runtime implementation: its neutral value 1.0 is
  accepted as a no-op, and any other value is refused with a 400, because a
  silently dropped sampling parameter serves output the client did not ask
  for.
- **Reasoning split.** Responses separate chain-of-thought from the answer:
  the assistant message carries `content` plus a `reasoning_content` field
  when the model emitted a think block (`chat_stream.ReasoningSplitter`
  incrementally, or `split_complete_text` on the non-streaming path).
- **IO edge**: `make_handler` builds a stdlib `BaseHTTPRequestHandler`
  (HTTP/1.1) with `GET /health` and `POST /v1/chat/completions`; it parses bytes,
  calls the core, writes bytes.
- **Concurrency**, a single-threaded `HTTPServer` with a `serve_lock`:
  `serialized_generator` / `serialized_stats` hold the lock across the whole
  fetch/generate/insert so the shared mutable model and the in-memory
  `PromptCacheStore` stay consistent; the store hands entries out by move, so
  that span is one critical section.

### Streaming responses (SSE)

The same endpoint streams when the request sets `stream: true`
(`request_stream_options` validates `stream` and
`stream_options.include_usage`; `stream_options` without `stream=true` is a
400). The response is `text/event-stream`, written by `_ChatSSEWriter`:

- One `chat.completion.chunk` event opens the stream with the assistant role;
  subsequent chunks carry incremental deltas. `ReasoningSplitter`
  (`chat_stream.py`) routes think-block text to `reasoning_content` deltas and
  answer text to `content` deltas as tokens arrive.
- During prefill, before the first token exists, the writer emits `: prefill`
  SSE comment lines as keep-alives so proxies and clients do not time out on
  long prompts.
- The terminal chunk carries the `finish_reason`; with
  `stream_options.include_usage` a final usage chunk follows (the same usage
  block as the non-streaming response); the stream ends with `data: [DONE]`.
- The 200 status and SSE headers are committed only after all request and
  context validation has passed: the generator invokes the transport's
  `ready_callback` after KV-policy and context-limit checks but before the
  cache fetch, so a validation failure is still a clean JSON error response,
  and a failed socket write cannot cost the session its chain entry.
- An error after the stream has started is sent as an in-stream `error` event
  and the connection closes; a disconnected client surfaces as
  `ClientDisconnected` and generation stops.

`http.serve` wires it together: load once, resolve thinking, prime generation,
build the in-memory cache generator (`build_cache_generator` →
`PrefixCacheGenerator`), expose `/health` stats, and run until interrupted
(clean shutdown closes the server and the cache generator). `/health` reports
status and model id, the `prompt_cache` block (formats, entry and byte counts,
plus the `disk` sub-block with the disk store's counters when disk KV is on),
and an `ssd_streaming` block with the streaming runtime's counters when the
package streams.

---

## Source map

| File | Role |
|---|---|
| `serve.py` | Load `(model, tokenizer, manifest)` from the manifest; `generate_with_metadata` / `generate_once`; `moespresso-generate` + `moespresso-verify` CLIs; runtime truth line. |
| `build.py` | Manifest-driven build via the proven jang loader; adapter selection; routed-expert install; mixed-bit wrapping; no dequant at load. |
| `generation.py` | Pure `GenerationResult` contract shared by serve + HTTP. |
| `http.py` | OpenAI-compatible HTTP: pure core + stdlib IO edge; the **single render site**; SSE streaming; startup warmup; `moespresso-serve` CLI. |
| `chat_stream.py` | `ReasoningSplitter`: incremental think-block/content splitting for streaming deltas. |
| `verify.py` | Pure, fail-closed manifest, identity, tensor-key, and sidecar gate; off the hot path. |
| `thinking.py` | Per-family thinking on/off resolution; refuse loudly. |
| `kv_policy.py` | Pure live-KV policy parse/validate → mlx_lm kwargs (in-memory). |
| `prefix_cache.py` | In-memory prefix reuse: `PromptCacheStore`, one live timeline per chain; declared-context-limit refusal. |
| `disk_kv.py` | The disk KV checkpoint tier, on by default when serving (`docs/disk_kv.md`). |
| `kquant_install.py` | Manifest-driven swap of constructed MLX modules to mlx-kquant module classes before K-quant wire bytes load. |
| `owned_switchglu.py` | `OwnedSwitchGLU` forward immune to jang's class-level fused patch (mixed gate/up bits). |
| `deepseek_v4/` | DeepSeek-V4 runtime graph adapter, cache contract, native/helper probes, and speed replay tools. |
| `qwen/` | Qwen-family runtime kernels: flash-style q8 full attention, prefill chunk planning, sorted SwitchGLU. |

Packages that carry an `agentic_profile.json` sidecar expose recorded
agent-loop defaults (tool dialect, repair, sampling) to agent clients;
`agentlib` reads it from the package directory. The sidecar is documented in
`docs/package_format.md`.
