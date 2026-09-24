# Disk KV cache: restart-warm and cross-session resume

The disk KV cache restores a served model's target prompt-cache prefix from
disk and prefills only the suffix. A long session that would otherwise
reprefill its whole history after a restart resumes warm, and a new session
whose prompt shares a long prefix with an earlier one (an agent client's fixed
system prompt and tool schemas) skips the shared region. Resumable DeepSeek-V4
DSpark requests may bind an optional drafter-state companion to that target
checkpoint. The target remains the authoritative cache entry.
`docs/speculative_decoding.md` states which drafter families are resumable and
what the runtime checks before it resumes speculation; this document covers the
durable side.

The store is single-process-per-root and narrow: it recovers an exact token
prefix into the same package and the same cache policy, and falls back to a
cold prefill on anything it cannot prove safe. A DSpark companion is held to
additional model and producer identity checks. Rejecting it does not reject a
valid target.

Serving enables the store by default under a per-package root in the user
cache directory; `MOESPRESSO_DISK_KV=off` turns it off. The in-memory prefix
cache is consulted first; the disk store is consulted only
on an in-memory miss.

---

## What it does

A served model keeps its prompt cache in memory. When the process exits, that
memory is gone, so the next process starts cold and reprefills the full prompt on
the first request. With the disk KV cache enabled, the running process writes an
aligned checkpoint of the live cache at configured token boundaries during prefill.
A later process, pointed at the same disk root, restores the longest checkpoint
whose token prefix matches the incoming request exactly, then prefills only the
tokens past that prefix. The restore reaches the first generated token faster than
a cold prefill of the same prompt.

For an eligible DSpark request, every new aligned speculative checkpoint may
add an optional companion to its authoritative target. The writer commits the
target cache first, then stores a capsule for the DSpark projected-history rings
and scalar state at the same frontier. Restore follows the same order. A
compatible pair continues speculative decoding from the suffix. If the
companion is absent, unavailable, corrupt, or incompatible, the target still
serves the suffix with plain decoding.

The restore is transparent to clients. A chat-completions client that resends its
full conversation history sends a prompt whose leading tokens match a stored
checkpoint; the server restores that prefix and generates from the new suffix. No
protocol change is needed.

The generic cache payload lacks next-token logits. An exact whole-prompt hit
therefore prefills the full prompt, reports `exact_fallback`, and skips
companion restore. Qwen4 KVarN stores the final logit row and can sample from
an exact whole-prompt checkpoint.

## Qwen4 KVarN4 checkpoints

The Qwen4 generation adapter shares the disk store, root lock, byte budget,
write-depth cap, and startup configuration. Its payload holds packed K4/V4
records, exact sink and live tail rows, compressed and pending attention index
state, three-axis positions, GDN recurrence and convolution state, and PLE
token and convolution history. Restore preserves the packed bytes and rebuilds
independent mutable storage. It does not run the KVarN quantizer or restore
process-local rollback markers.

Checkpoint identity covers the package, cache-routing configuration, rendering
identity, packed layout, and composite-state schema. Every mixer and PLE state
must share the selected token frontier. The shared store refuses incompatible
or invalid payloads and falls back to cold prefill.

The memory tier is consulted first. Qwen4 captures memory snapshots at the end
of prefill. A follow-up request processes previous generated text in its
suffix, preserving unbiased prefill even when decode uses cache-biased routing.
Memory entry and byte limits still apply. Disk snapshots occur only at aligned
prefill frontiers, including an aligned full prompt; decode writes none. Each
snapshot stores the final logit row, so an exact prompt hit can sample without
another prefill. Usage reports the uncached suffix in `prompt_tokens` and the
restored prefix in `prompt_tokens_details.cached_tokens`.

Retained snapshots consume memory outside the expert pool. Under a constrained
memory budget, cap that tier with `--prompt-cache-bytes`; the separate disk byte
budget does not limit it.

The KV payload excludes expert-pool residency. With cache-biased routing, a
fresh process or skipped prefill may produce a different resident expert set.
Numerical restore comparisons must control that set or use original routing.
The cache bias and expert loader remain unchanged. These checkpoints do not
enable Qwen MTP or publish speculative state.

---

## Configuration

Serving turns the store on by default with bounded defaults; every value can
be overridden through environment variables read once at startup.

- `MOESPRESSO_DISK_KV`: the mode. `off` (or `0`) disables the store for the
  process. `frontier` requests it explicitly, which makes configuration
  faults refuse startup instead of degrading (see below). Unset means the
  serving default: enabled with the derived root and default stride and
  budget. Any other value refuses startup.
- `MOESPRESSO_DISK_KV_ROOT`: the disk root directory. Default:
  `$XDG_CACHE_HOME/moespresso/disk_kv/<package-fingerprint>` (falling back
  to `~/.cache`). The fingerprint keys on the package directory so servers
  for different packages never contend for one root lock; correctness never
  depends on the split, because the checkpoint scope gates every restore.
  One process owns a root for its lifetime (see the root lock below).
- `MOESPRESSO_DISK_KV_STRIDE`: the checkpoint stride in tokens. Default
  1024; must be a positive multiple of 256. A checkpoint is written each
  time prefill reaches a multiple of the stride. Checkpoints are written
  only during prefill and only for frontiers not already on disk, so the
  write cost lands once, in the first request that covers a new prefix
  region. The request usage block reports confirmed write counts, and the
  blocking work remains included in both generation-local first-token latency
  and ready-to-first-token latency. Decode never writes.
  A smaller stride shortens the re-prefilled tail after a restore
  (at most one stride, for divergence points within the write-depth cap;
  past the cap the unsaved tail re-prefills whole) at the cost of more
  first-time writes.
- `MOESPRESSO_DISK_KV_BYTES`: the byte budget for stored payloads, per
  root, not machine-wide: every package fingerprint has its own root and
  its own budget, and quarantined payloads sit outside it. Target payloads
  and optional model-state companions share this budget. Default 8 GiB
  when serving (a checkpoint set covering one long agent prompt runs to a
  few GiB, so the default holds a handful of hot prefix regions); the
  literal `unlimited` disables eviction. A positive value caps the
  payload bytes on disk and evicts least-recently-used checkpoints; the
  payload is written before eviction runs so its exact size is known,
  which means the store can briefly exceed the budget by one payload
  during a write. The budget bounds retention, not write traffic; the
  write-depth cap bounds that. A value of `0` refuses startup with a
  message pointing at `MOESPRESSO_DISK_KV=off`, because a zero budget
  cannot hold any checkpoint. A negative value refuses startup as a
  misconfiguration.
- `MOESPRESSO_DISK_KV_WRITE_DEPTH`: the checkpoint write-depth cap in
  tokens. Default 16384 when serving; the literal `unlimited` writes
  frontiers at any depth. Each checkpoint is a complete snapshot, so
  written bytes grow quadratically with region depth, while cross-session
  restores land in the shallow shared-prefix region (an agent client's
  system prompt and tools). The cap keeps the whole shared-prefix benefit
  and drops the deep-tail write traffic of a long conversation. The same
  frontiers gate target and DSpark-companion capture. Restores are
  unaffected: the read path serves whatever checkpoints exist.

Startup failure policy: with `MOESPRESSO_DISK_KV=frontier` set, a root that
cannot open (already locked, unwritable) refuses startup, as does any
malformed value. Under the serving default, an unopenable store prints one
`[serve] disk_kv=off (reason)` line and the process serves memory-only: a
locked cache directory must not take down a server nobody configured for
disk KV. Malformed explicit values (a bad stride or budget) always refuse.

## On-disk footprint and removal

The default location keeps everything under one directory:
`~/.cache/moespresso`. Growth is bounded by the byte budget per package
root (8 GiB by default), enforced by least-recently-used eviction. Target
entries retain the v1 `index.json` and `payloads/` layout. Optional companions
use their own `attachments/index.json`, `attachments/payloads/`, and
`attachments/quarantine/` paths. This keeps target-only roots readable and
allows attachment faults to stay isolated. There is no free-disk-space probe:
the budget is the bound, and a write that fails for any reason, a full disk
included, is skipped and logged while the request completes normally. Deleting
the directory (or any single root) at any time is safe: the server holds no
assumption that a checkpoint survives, and a missing or mismatched entry means
cold serving,
never a wrong restore. Package managers do not remove user caches on
uninstall, so after removing MoEspresso itself, `~/.cache/moespresso` is
the one path to delete.

---

## The frontier rule

A frontier is a token count where every layer's cache sits at a clean boundary at
the same time. All cache families align on the MLX 256-token cache step, so a
frontier is a multiple of 256; the DeepSeek-V4 composite cache satisfies both of
its compression ratios at 256, and the Qwen hybrid's recurrent state carries no
alignment concern. The configured stride must be a multiple of that 256-token step,
validated at startup. A checkpoint is written only when the live cache is exactly at
a frontier: the writer plans the prefill as full-size chunks with one shorter chunk
ending exactly on each frontier, so the prefill callback fires on every stride
boundary regardless of the restored prefix length, and the cache classes' own
reported offsets confirm the position before a byte is written. A cache that is not
exactly at the frontier refuses the write, so a checkpoint that describes a token
count it does not hold is structurally impossible. A single uniform step cannot do
this: it must divide gcd(first_gap, stride), which collapses to a few tokens when
the restored prefix is not stride-aligned and makes long prefills unserviceable.

DSpark uses a speculative-only prefill plan with the same absolute frontier
allowlist. Its callback runs after the target forward, DSpark tap ingest, and
materialization. It independently checks every target cache and the drafter
state against the same frontier before it copies anything. The callback never
runs during the final anchor forward, verify, or decode. A target write or
budget refusal stops the companion write at that frontier. A companion failure
leaves the committed target intact and does not stop later target captures.

---

## What is promised, and what is not

Promised:

- A restore recovers exactly to the last completed checkpoint. A prompt whose prefix
  passes a stored frontier but stops short of the next one restores to that frontier
  and prefills the rest. Tokens generated past the last written frontier are not on
  disk.
- The restore is exact-prefix only. The stored token prefix must equal the leading
  tokens of the request bit for bit. There is no rounding a key down to a nearby
  checkpoint and no byte-prefix or fuzzy match.
- A checkpoint restores only into the same package, the same rendering, and the same
  KV policy that wrote it. The safety key is the serve cache scope (package, rendering
  identity, live KV format, group size, quantized KV start, cache payload kind) joined
  with the cache-class layout and the disk schema version, plus the token-prefix hash
  and the prefix length. Any mismatch fails closed to a cold prefill. Exactly one
  cache-class disagreement is converted instead of refused: under a quantized live-KV
  policy the KV layers convert from the raw cache class to the quantized one once the
  offset passes the policy threshold, so an aligned save on such a session records the
  quantized class while a fresh cache starts raw. The restore converts the fresh,
  empty cache to the recorded class before grafting (a stateless conversion; the
  grafted state carries the recorded offset, group size, and bits). Every other class
  disagreement still refuses.
- A DSpark companion is accepted only after its target checkpoint restores.
  Its content identity binds the target cache id, scope hash, token count,
  token-prefix hash, attachment kind and schema, drafter family, sidecar artifact
  id, capsule kind and schema, and speculative producer rail. The producer rail
  includes the resolved draft schedule. The imported capsule and all target
  caches must report the target frontier before speculative generation resumes.
- A corrupt, truncated, or missing payload fails closed. The load raises before any
  cache reaches the model, the entry is quarantined, and the engine continues on cold
  serving.
- A selected DSpark companion whose payload or model import is invalid is
  quarantined independently. An identity-incompatible companion is not selected
  and appears as missing for the current request. The already validated target
  remains indexed and serves plain suffix generation. A missing companion
  reports `missing`; a companion-index fault reports `unavailable`; a rejected
  payload or model import reports `invalid`.

Not promised:

- No portability across models or machines. A checkpoint is meaningful only to the
  package and layout that wrote it.
- No recovery of decode-time state past the last prefill frontier. Capture is
  prefill-time in this version.
- The generic payload lacks next-token logits and takes `exact_fallback` on a
  whole-prompt hit. Qwen4 KVarN includes its final logit row, as described above.
- No additional file compression, no trim-back to an unaligned length, or concurrent
  cross-process sharing of a root.

### The cost of crossing an unwritten frontier

Aligning prefill to the stride splits a prompt that a single default step would
have prefilled in one chunk into stride-sized chunks. Each crossed frontier
writes a payload under the serve lock before the first token, so both latency
fields include it. On the reference audit prompt (3844 tokens, stride 2048),
the writer fired once at token 2048 with a 386 MB payload, and the blocking
write cost 0.135 s. Generation-local first-token latency with the writer on was
18.79 s against 16.82 s with it off. The 0.135 s write is the marginal blocking
cost; the rest of the gap is the prefill chunk geometry that frontier capture
requires. A later process restoring that checkpoint reaches its first token
faster than a cold prefill of the same prompt because only the suffix is
prefilled.

---

## Operational notes

- Root lock, single owner. One process owns a disk root through a non-blocking file
  lock acquired before the model load. A second process pointed at the same root is
  refused loudly at startup rather than waiting or stealing the lock. This is why the
  index is a single JSON file rewritten atomically under the lock: the store is
  single-process by contract.
- Budget and eviction. Under a byte budget and a readable companion index, a
  write that would exceed the cap first evicts least-recently-used DSpark
  companions, then target checkpoints (by last-used time, then creation time)
  until the new payload fits. Evicting a target removes its dependent
  companions first. An incoming companion cannot evict its parent target; it is
  skipped when the parent plus companion cannot fit. If the companion index is
  unavailable, physical companion files still count toward the cap. Target
  writes that fit without eviction continue; a write that would require
  dependency-aware eviction is skipped. A payload that alone exceeds the whole
  budget is skipped and logged.
- Startup cleanup. After acquiring the lock the store deletes leftover temp payloads
  and orphan payloads (payload files no index entry references) left by a crashed
  previous owner. It also removes companion entries without an exact target and
  companion payloads without an index entry. The quarantine directories hold
  payloads that failed a load check; their aging is left to the operator.
- Write faults disable the writer for the request. A hard failure during a
  checkpoint write (a full or failing disk) logs one line and stops further
  write attempts for that request, because each later frontier would
  serialize an even larger snapshot into the same fault. The request itself
  completes normally; the next request tries again.
- Index faults disable writes until restart. A confirmed index fault (a
  corrupt index file or a failing index write, discovered by any index
  access: writer planning, a restore lookup, the LRU touch, a checkpoint
  write, or the health snapshot) logs one line, deletes any finished
  payload the fault interrupted, and stops checkpoint writes for the
  store's lifetime, with the writer refusing before payload serialization,
  so a broken index cannot cost payload serializations on later requests.
  Restores keep falling back to cold serving, and the `/health` disk block
  reports the fault without failing the endpoint: `error` when the index
  is unreadable, and `writes_disabled` with `writes_disabled_reason`
  whenever writes are off. A quarantine that cannot mutate the index (a
  readable index on an unwritable disk) is itself a confirmed fault: the
  store disables writes, keeps the caller's original restore error, and
  remembers the entry as dead so later requests never re-load the
  rejected payload while the next valid checkpoint still restores. A
  payload move failing after a successful index removal is housekeeping,
  not an index fault: writes stay enabled and the leftover file is an
  orphan the next open cleans up. Reopening the store (a server restart)
  retries; startup cleanup removes any orphaned payloads.
- Attachment faults stay local. Failure to open or read the companion index
  disables companion reads and writes until restart and records
  `attachments_unavailable_reason`. It does not disable target restores or
  target writes that fit the conservative physical-byte accounting. Companion
  quarantine removes only the companion index row and payload. Target eviction
  remains authoritative and cascades removal to every dependent companion when
  that mapping is readable.
- Visibility. Operator logs record target and companion writes, skips, hard
  failures, restores, and quarantines. An absent companion is reported in the
  request without adding a log line. The `/health` endpoint's
  `prompt_cache.disk` block reports the
  combined `payload_bytes`, separate `target_payload_bytes` and
  `attachment_payload_bytes`, `attachment_entries`, `attachments_available`,
  and separate target and attachment restore, write, eviction, and quarantine
  counters. `attachment_accounting` reports `index`, `filesystem`, or `unknown`;
  counts or bytes that cannot be established are `null`. An attachment fault
  adds `attachments_unavailable_reason`.
  Request `usage.prompt_cache.event` continues to report the target outcome,
  including `disk_hit`. `usage.prompt_cache.drafter_state.event` independently
  reports `hit`, `missing`, `invalid`, or `unavailable` when a DSpark companion
  was consulted. A valid target payload reports `disk_restore_seconds`; an
  exact payload that cannot supply continuation logits still reports its
  reconstruction cost beside the `exact_fallback` event. A successful DSpark
  companion restore adds `drafter_state.restore_seconds`, covering disk-layer
  load and validation plus the serve preflight import that creates usable live
  state. Failed or absent companions omit that duration.
  `disk_checkpoints_written` and `disk_drafter_states_written` report confirmed
  writes separately. The
  corresponding `disk_checkpoint_write_seconds` and
  `disk_drafter_state_write_seconds` arrays report each blocking write duration,
  rounded to six decimal places. Empty timing arrays are omitted.
