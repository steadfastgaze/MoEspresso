# Disk KV cache: restart-warm and cross-session resume

The disk KV cache restores a served model's prompt-cache prefix from disk and
prefills only the suffix. A long session that would otherwise reprefill its
whole history after a restart resumes warm, and a new session whose prompt
shares a long prefix with an earlier one (an agent client's fixed system
prompt and tool schemas) skips the shared region. The store is
single-process-per-root and narrow: it recovers an exact token prefix into
the same package and the same cache policy, and falls back to a cold prefill
on anything it cannot prove safe.

Serving enables the store by default under a per-package root in the user
cache directory; `MOESPRESSO_DISK_KV=off` turns it off. The in-memory prefix
cache is unchanged and is consulted first; the disk store is consulted only
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

The restore is transparent to clients. A chat-completions client that resends its
full conversation history sends a prompt whose leading tokens match a stored
checkpoint; the server restores that prefix and generates from the new suffix. No
protocol change is needed.

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
  region, and is reported in that request's usage block. Decode never
  writes. A smaller stride shortens the re-prefilled tail after a restore
  (at most one stride) at the cost of more first-time writes.
- `MOESPRESSO_DISK_KV_BYTES`: the total byte budget for stored payloads per
  root. Default 32 GiB when serving; the literal `unlimited` disables
  eviction. A positive value caps the payload bytes on disk and evicts
  least-recently-used checkpoints before a write that would exceed it. A
  value of `0` refuses startup with a message pointing at
  `MOESPRESSO_DISK_KV=off`, because a zero budget cannot hold any
  checkpoint. A negative value refuses startup as a misconfiguration.

Startup failure policy: with `MOESPRESSO_DISK_KV=frontier` set, a root that
cannot open (already locked, unwritable) refuses startup, as does any
malformed value. Under the serving default, an unopenable store prints one
`[serve] disk_kv=off (reason)` line and the process serves memory-only: a
locked cache directory must not take down a server nobody configured for
disk KV. Malformed explicit values (a bad stride or budget) always refuse.

## On-disk footprint and removal

The default location keeps everything under one directory:
`~/.cache/moespresso`. Growth is bounded by the byte budget per package
root, enforced by least-recently-used eviction. Deleting the directory (or
any single root) at any time is safe: the server holds no assumption that a
checkpoint survives, and a missing or mismatched entry means cold serving,
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
- A corrupt, truncated, or missing payload fails closed. The load raises before any
  cache reaches the model, the entry is quarantined, and the engine continues on cold
  serving.

Not promised:

- No portability across models or machines. A checkpoint is meaningful only to the
  package and layout that wrote it.
- No recovery of decode-time state past the last prefill frontier. Capture is
  prefill-time in this version.
- No compression, no trim-back to an unaligned length, no cross-process sharing of a
  root.

### The cost of crossing an unwritten frontier

Aligning prefill to the stride splits a prompt that a single default step would have
prefilled in one chunk into stride-sized chunks, and each crossed frontier writes a
payload under the serve lock inside first-token latency. On the reference audit prompt
(3844 tokens, stride 2048) the writer fired once at token 2048, a 386 MB payload, and
the blocking write cost 0.135 s. First-token latency with the writer on was 18.79 s
against 16.82 s with it off; the 0.135 s write is the marginal blocking cost and the
rest of the gap is the prefill chunk geometry that frontier capture requires. A later
process restoring that checkpoint reaches its first token faster than a cold prefill
of the same prompt: the restore plus the suffix prefill runs in less time than a full
cold prefill because only the suffix is prefilled.

---

## Operational notes

- Root lock, single owner. One process owns a disk root through a non-blocking file
  lock acquired before the model load. A second process pointed at the same root is
  refused loudly at startup rather than waiting or stealing the lock. This is why the
  index is a single JSON file rewritten atomically under the lock: the store is
  single-process by contract.
- Budget and eviction. Under a byte budget, a write that would exceed the cap first
  evicts least-recently-used checkpoints (by last-used time, then creation time) until
  the new payload fits. A payload that alone exceeds the whole budget is skipped and
  logged; the store never evicts every other entry to make room for one oversized
  payload.
- Startup cleanup. After acquiring the lock the store deletes leftover temp payloads
  and orphan payloads (payload files no index entry references) left by a crashed
  previous owner. The quarantine directory holds payloads that failed a load check;
  its aging is left to the operator.
- Visibility. One operator log line prints on the serve stderr for each checkpoint
  decision (write, skip with reason, restore, quarantine). The `/health` endpoint's
  `prompt_cache` block carries a `disk` sub-block with the store's counters since
  startup: enabled, root, stride, entries, payload and budget bytes, and the restore,
  write, eviction, and quarantine counts. A request that restored from disk reports the
  `disk_hit` event in its `usage.prompt_cache` block, and a request that wrote frontier
  checkpoints reports how many.
