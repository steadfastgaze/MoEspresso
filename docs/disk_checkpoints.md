# Native prompt checkpoints

## Single slot

`DiskCheckpointSlot` provides one checkpoint slot in an explicitly supplied
private directory. It is not enabled by the command-line server by default.
`DiskLimits` requires a maximum file size and an I/O staging allowance.
Library callers can open a slot directly with an explicit root and budgets.

The slot accepts any positive committed frontier, including short prompts,
partial prefill chunks and consumed generated tokens. Saved state includes token IDs,
last-row logits, PLE, GDN recurrence and convolution, QSA index state, and KVarN
records, sinks and physical tail rings. Unused sink and tail rows are omitted;
restoration preserves head strides and wrapped physical positions. Expert
residency is not saved or reset. Sampled tokens not yet consumed by the engine
are excluded and must be evaluated when they appear in a later request.

Compatibility binds the actual parsed package manifest and tokenizer,
generation configuration and chat-template assets, supported state version and
model geometry. Executable contents, device identifiers, paths, OS version,
expert capacity and ordinary GPU prefill width do not determine compatibility. A checkpoint
must fit the receiving engine's context limit. Changed residency or chunk width
may affect subsequent outputs without making the committed state invalid.
Package payloads must remain immutable; manifest identity is not
independent verification of weight bytes. This format does not read MoEspresso
3 checkpoints.

Combined ANE execution additionally binds its numerical producer and physical
program width to cache identity. GPU and GPU+ANE execution cannot interchange
checkpoints. An incompatible producer returns no match; changing its identity
does not convert saved state.

`save` validates independently committed component frontiers and size admission.
It streams typed tensor spans and a checksum to a temporary
file, syncs that file, atomically replaces the slot, then syncs the directory.
An exclusive nonblocking lock prevents concurrent writers. Fixed leaf names,
private ownership and descriptor-relative operations reject unsafe filesystem
entries. A failure before rename preserves the old slot. A directory-sync
failure after rename reports uncertain durability without deleting the new
file or claiming rollback.

`load_match` checks file and section geometry, scope, exact requested token
prefix, checksum and state semantics before binding a candidate to the current
engine. It returns a fresh session without modifying the live expert pools.
An absent slot or incompatible scope/prefix returns no match. Malformed files,
I/O and allocation failures return errors for the caller to handle. No corrupt
or partially restored session is published.

Save and restore use a staging allocation capped at 8 MiB, independent of
checkpoint length. Restore fills an unpublished device-state candidate directly;
no complete host snapshot is allocated. Candidate token IDs become its retained
lineage. Atomic replacement temporarily retains old and new files, so the limits
are not a peak filesystem or process RSS guarantee. The HTTP integration uses
the indexed store below for retention and restart lookup.

## Indexed retention

`DiskCheckpointStore` retains multiple prompt prefixes using the same native
payload format and compatibility scope. The serialized HTTP owner inspects
memory first. A complete memory hit needs no disk lookup; a partial hit checks
for a deeper disk prefix before allocating restored state. Memory wins equal
frontiers and remains available if the deeper disk candidate is damaged.
Runtime cache faults disable that tier
for the request and serving continues from another tier or cold state.
The default `prefix` policy writes bounded prefill anchors. The `full` policy
also publishes completed prompts and committed generated frontiers.

`StoreLimits` requires a retained payload byte budget, an entry limit from 1
through 64, and the per-file and staging limits in `DiskLimits`. A bounded,
versioned JSON index records exact scoped prefix identities, payload lengths
and monotonic LRU recency. Filenames are generated from 65 fixed slots, including
one publication spare; the index cannot supply arbitrary paths.

The store restores the longest matching prefix after checking index metadata
against the payload, then checking the payload's exact tokens, checksum and
state. Missing or deterministically invalid entries are excluded so a shorter
valid checkpoint can be tried. Transient I/O, allocation and live restore
failures retain the payload and disable further restores for that store
instance. A failed LRU update does not invalidate an already restored session.

Writes deduplicate an existing scoped prefix. A new payload is synced and
published before its index entry. In `prefix` mode, earlier anchors remain
eligible for reuse by conversations that diverge before a deeper checkpoint.
Ordinary LRU eviction enforces the byte and entry limits. In `full` mode, an
advancing token history keeps the incoming frontier and its nearest two saved
ancestors. Superseded ancestors are removed before ordinary LRU eviction so
old turns do not crowd out independent chats.
Eviction commits the reduced index before
removing victims. A directory-sync failure after publication preserves the
potentially referenced payloads and disables later mutations, including LRU
touches and invalid-entry cleanup. It reports uncertain durability, not rollback.

Opening a valid index reconciles missing or size-mismatched payloads, applies
lower configured retention limits, and removes only unindexed generated leaves
and temporary files. A missing index alongside payloads, or malformed metadata,
fails without treating those files as disposable orphans. Unrelated root files
are not removed. The store shares the single-slot component's exclusive root
lock and private descriptor-relative filesystem checks.

The byte budget bounds indexed retention. Atomic publication can temporarily
retain an additional payload, and interrupted cleanup can leave bounded orphan
files until the next successful open. Save and restore have the same bounded
staging allocation as the single slot. These limits do not bound
peak filesystem use or process RSS.

## Managed aggregate retention

Default serving shares its retained byte allowance across recognized roots below
the selected `moespresso` user-cache directory: native `v4/disk_kv/<64 hex>`
and legacy `disk_kv/<12 hex>`. Discovery counts actual generated payload sizes,
including known temporary and quarantined files and legacy companions. Legacy
checkpoint bytes are never imported. Each native package root still has its own
64-entry limit.

Startup and admission of a new checkpoint reclaim the least recently used
inactive roots until the current store can fit. Index modification time orders
roots; the active store uses its entry-level LRU when a smaller allowance remains.
An aggregate lock serializes native admission through publication and cleanup.
A busy aggregate lock refuses that admission; serving continues with available
state. Each cache root also keeps its existing native or legacy lock. Active
roots and unrecognized indexes consume the observed allowance and are preserved.
When they leave insufficient space, the current store shrinks its own retention
and declines new files that cannot fit. Concurrent legacy servers can grow
independently, so their active files may temporarily keep total usage above the
configured allowance.

Cleanup pins owned directories and generated leaves, rejects symlinks and hard
links, and publishes empty indexes before removing inactive payloads. Unrelated
files and directories are preserved. The reserved namespaces above are managed
storage. An explicit root outside them has its own byte allowance and is never
discovered by aggregate cleanup. Cleanup can be logged as `disk_cleanup` with
the reclaimed byte count. Evicted state can be recreated through cold inference.

## Serving configuration

Serving enables disk reuse by default with automatic capacity admission. An
explicit `--max-memory-gb` caps the same automatic component budget and does
not disable the disk tier. The advanced fixed-capacity `--expert-capacity`
diagnostic path disables memory and disk cache reuse.
The root is `XDG_CACHE_HOME/moespresso/v4/disk_kv/<manifest-digest>`, or
`HOME/.cache/moespresso/v4/disk_kv/<manifest-digest>` when XDG is unset or
empty. The digest is SHA-256 of the package manifest contents, so moving an
unchanged package does not change its cache namespace. This native
namespace does not share MoEspresso 3 cache files; explicit roots must also be
dedicated native directories. Missing directories are created with mode 0700;
existing directories are never chmodded or repaired.

Use `moespresso serve PACKAGE --no-disk-cache` to disable disk reads and writes.
This avoids accessing the cache root without deleting existing checkpoints.
RAM caching is independent; add
`--prompt-cache-gb 0` to disable both tiers.

| Option | Default | Accepted values |
| --- | --- | --- |
| `--disk-cache-mode MODE` | `prefix` | `prefix` writes shallow prefill anchors; `full` also saves completed prompts and committed generated state |
| `--disk-cache-dir PATH` | Native user-cache root | Dedicated private directory, relative to the working directory or absolute |
| `--disk-cache-gb GIB` | 8 | Positive finite retention budget in GiB, across managed roots or within an explicit root |

`--no-disk-cache` conflicts with the disk-cache options. An unlimited retention
budget is not supported. In default `prefix` mode, writes occur only during
prefill, at completed full-chunk frontiers crossing a 1024-token stride, up to
16384 tokens. The end of a prompt or answer does not trigger an extra write.
An already stored exact prefix is not written again. This targets shared
harness prefixes without identifying semantic system-message boundaries.

`full` mode adds completed prompts and committed generated frontiers, including
short prompts and state beyond the anchor write depth, subject to context and
retained byte limits. These writes block inference; prompt saves delay first
token delivery and final saves delay request completion. Both modes use the
same payload format and can read existing compatible checkpoints, including
deeper checkpoints previously written in `full` mode. Selecting `prefix` does
not delete them. Neither the stride nor the depth limits conversation length.

RAM retention is independent of disk policy. With `prefix` mode, a restart,
eviction or snapshot larger than the RAM budget requires evaluating the tail
after the best saved prefix. A snapshot that repeatedly exceeds that budget
can therefore cause repeated tail prefill between turns. Choose `full` mode
when deeper disk resume is worth its write cost, or increase `--prompt-cache-gb`
when host memory permits. Short conversations remain reusable in RAM even
when they never reach a disk anchor.

Writes occur after completed prefill chunks cross a stride boundary,
including when a restored prefix starts partway through a stride. The server
logs the first cold-prefill anchor. A 2048-row engine first saves an anchor at
token 2048. Chunks are never split to reach a disk checkpoint.
The maximum file allowance is the smaller of the
retained payload budget and 8 GiB; it is independent of staging memory.

Any explicit disk-cache option makes an unavailable root fatal at startup. With
no disk-cache options, an unavailable default location, directory or store emits
a warning and serves without disk caching. Invalid budgets fail before model
loading. `--prompt-cache-gb 0` disables only memory retention; use
`--no-disk-cache` separately. Fixed
`--expert-capacity` bypasses admission and disables both caches. Explicitly
configured disk settings conflict with fixed capacity; `--no-disk-cache` is allowed.

## HTTP cache reporting

The `/health` response reports the active serve conditions, including expert
capacity, maximum context, prefill capacity, whether adaptive memory is active,
the decode routing mode and cache-tier availability:

```json
{
  "status": "ok",
  "model": "<package-artifact-id>",
  "expert_capacity": 178,
  "max_context": 131072,
  "prefill_capacity": 512,
  "adaptive_memory": true,
  "cache_routing": "prefer-resident",
  "prompt_cache": {
    "memory_enabled": true,
    "disk_enabled": true,
    "disk_mode": "prefix",
    "retained_bytes": 1073741824
  }
}
```

`disk_mode` is `prefix` or `full` while disk caching is active, and null when
disabled or unavailable. The retained byte count can be null when it cannot be
read. These fields report the current engine and configured cache tiers; they
do not report system pressure or power use.

Chat completion usage reports cache evidence for each request. The standard
`usage.prompt_tokens_details.cached_tokens` field gives the number of prompt
tokens restored from either tier. The `usage.prompt_cache` object includes:

| Field | Meaning |
| --- | --- |
| `event` | `cold_start`, `memory_hit`, `disk_hit` or `disabled`, describing the request's initial cache lookup. |
| `cached_tokens` | Restored prompt tokens, matching `prompt_tokens_details.cached_tokens`. |
| `disk_checkpoints_written` | Number of successful disk checkpoint writes during the request. |
| `last_disk_checkpoint` | Token frontier of the last successful write in this request; zero when none was written. |
| `disk_checkpoint_write_seconds` | Total time spent writing checkpoints during the request. |
| `capacity_changed` | Whether adaptive serving changed expert capacity during generation. |

Disk write and restore failures are logged and fall back to available cache
tiers or cold prefill. Expert-capacity changes preserve committed conversation
state in both tiers. They may change later residency-biased routing, but do not
require replaying a cached prefix. The response fields describe the operations
that did occur; prefix mismatch, eviction and size limits can still cause misses.

The CLI fixes indexed entries at 64 and I/O staging admission at 8 MiB.
Capacity planning charges that staging buffer separately from memory
retention. The checked host-cache overlap is their sum; the ordinary session
component covers restored live state. A root failure after admission does not
resize expert pools, so its reserve may remain unused. Disk budgets do not
bound temporary filesystem peak, process RSS or later external memory pressure.
