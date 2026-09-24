# Pooled routed-expert runtime

The pooled runtime serves routed Mixture-of-Experts weights from persistent
per-layer MLX buffers. Their capacity is stable during each request. The
non-routed core stays resident. When every expert fits, all rows are loaded
before serving and inference performs no demand I/O. At smaller capacities,
missing routed rows stream from the package into the same buffers as routing
selects them. Full residency and bounded SSD streaming are two capacity points
of one graph. After a request, selected layers can replace their buffers with
larger ones through the adaptive transaction described below.

The design follows a hard rule: there is no Python or NumPy expert tensor
compute on the per-token path. The host performs routing bookkeeping and moves
encoded bundle bytes into persistent slots. Some codecs copy stored components
directly; IQ_K deinterleaves a stored `blocks` row into kernel-native streams.
The host does not dequantize weights or compute expert activations. All expert
math runs in MLX and Metal kernels, and one pooled SwitchGLU operation owns each
routed layer.

Source: `src/moespresso/runtime/`.

The Qwen4 full512 adapter uses cache-conditioned routing by default with
bounded expert pools. During decode it preserves the two strongest original
routes and gives resident candidates a factor-two preference for the remaining
positions. Prefill is unbiased. Full-resident Qwen4, DeepSeek-V4 and Ornith use
original routing. `--cache-routing off` disables Qwen4's preference. See
[Cache-Prior routing](cache_prior.md) for the algorithm and its storage behavior.

---

## 1. Component map

| File | Role |
| --- | --- |
| `ssd_streaming_build.py` | Build entry point: take the run lock, build the index, compute capacity, install pooled SwitchGLUs, load the non-routed core resident, seed cold-start residency. |
| `expert_index.py` | Per-layer bundle byte-offset index; maps `(layer, expert[, projection, component])` to an exact byte range in a shard. |
| `expert_loader.py` | Single-expert byte-range loader (`pread` one expert into a fresh `mx.array`); proof/test helper that uses the same primitive the pools use. |
| `pread_into.py` | Read a file byte range straight into a writable MLX buffer with `os.preadv` (no Python `bytes`), using a bounded fd cache. |
| `expert_slot_pool.py` | The persistent slot pool per (layer, projection); LFU residency, miss loads, hotness decay, transactional growth, and the on-device slot table. |
| `expert_pool.py` | Compact-pool proof primitive that validates bit identity between a remapped compact `packed` tensor and the full stack. The product hot path uses the persistent slot pools. |
| `expert_locality.py` | Pure router-locality analysis: activation histogram, per-layer hotlist, simulated LRU hit-rate curve. |
| `streaming_capacity.py` | Capacity math: byte cost per capacity unit from the index, memory budget, `choose_capacity`. |
| `pooled_switchglu.py` | The SwitchGLU forward over pooled experts, including codec-specific full-resident and bounded-residency scheduling. |
| `routed_decode_kernel.py` | MoEspresso-owned single-dispatch Metal kernel for the source-MXFP4 routed MLP on one decode token. |
| `native_gate.py` | Loader for the optional native MTLSharedEvent gate extension; falls back transparently if absent. |
| `streaming_run_lock.py` | Single-owner process lock so only one real-model streaming run executes at a time. |

---

## 2. The expert byte index (`expert_index.py`)

A package stores one routed layer's experts as a single **bundle** tensor
`...switch_mlp.experts.tq_bundle`, a `uint8 [n_experts, row_bytes]` array whose
row `e` concatenates that expert's projection components in declared order.
MXFP4 rows contain `packed`
and `scales`, K-quant rows contain their wire components, and IQ_K projections
contain one `blocks` component. The within-row geometry carries each
component's offset, byte count, shape, dtype, codec, and codec parameters in the
shard's safetensors metadata.

`build_expert_index(package_dir)` scans every shard's headers for bundle tensors,
pairs each with its layer's metadata geometry, and records absolute byte offsets.
It reads only headers and metadata JSON, never weight bytes, never MLX, never
jang. The resulting `ExpertIndex` answers:

- `locate(layer, expert, projection, component)` -> exact `ExpertByteRange`
  (`shard`, absolute `offset`, `nbytes`, `shape`, `dtype`) for one component.
- `locate_row(layer, expert)` -> the whole bundle row as one range, so a missed
  expert costs **one** `pread` instead of six.
- `row_components(layer)` -> the within-row slice geometry (offsets relative to
  the row start) for splitting a staged row into its components.
- `geometry(layer, projection)` -> `out_features`, `packed_cols`, `bits`, dtypes.

`validate()` performs cheap structural checks (rows tile the tensor exactly,
components tile each row in codec order without padding, first/last rows stay
in bounds) and returns a list of problems. Expert-count or component-geometry
mismatches fail validation.

---

## 3. Direct byte-range loading (`pread_into.py`, `expert_loader.py`)

The miss-loader must not materialize a Python `bytes` object or rebuild MLX
arrays from copied host data. MLX arrays expose a writable C-contiguous buffer,
and `os.preadv` writes a file byte range directly into a `memoryview` slice.

`pread_into(dst, path, *, file_offset, nbytes, dst_offset)` reads exactly
`nbytes` from `path:file_offset` into `dst` (an `mx.array` or any writable
C-contiguous buffer) at `dst_offset`, looping on short reads and raising
`PreadIntoShortRead` if the file ends early. It validates that the destination
view is writable, contiguous, and large enough. `os.preadv` is required;
`PreadIntoUnavailable` is raised otherwise.

`PreadFileCache` keeps a bounded LRU of read-only file descriptors so
high-frequency misses do not re-`open`/`close` the shard each time. It supports
refcounted `acquire_fd` (so an fd in use is never evicted) and is thread-safe.
`pread_into_cached` / `pread_view_cached` read through the shared default cache.

`expert_loader.load_expert(...)` is the single-component form: allocate a fresh
`mx.array` of the declared shape and dtype from the index and `pread` the bytes
in. The payload stays in its encoded format; no expert dequantization happens at
load time. This module is a proof/test helper. The product miss path uses the
same `pread_view_cached` primitive to fill persistent pool slots in place.

---

## 4. Capacity math (`streaming_capacity.py`)

Capacity is derived from the memory contract. One **capacity unit** means
"one resident expert slot in every routed (layer, projection) pool", so its byte
cost is summed across the whole model.

- `bytes_per_layer_slot(index)`: for each routed layer, the declared component
  bytes of one expert summed over its projections, read from the index geometry.
- `bytes_per_capacity_unit(index)` = the sum of those over all layers: the
  marginal RAM cost of raising capacity by one slot everywhere.
- `non_routed_payload_bytes(package_dir)`: header-only sum of every non-bundle
  payload tensor; this is the resident base the runtime keeps at startup.

`CapacityBudget` carries `available_bytes`, `resident_base_bytes`, `runtime_resident_bytes`,
`kv_activation_allowance_bytes`, `safety_margin_bytes`, `bytes_per_capacity_unit`,
`min_capacity`, `max_capacity`. Its `usable_bytes` subtracts the model base,
separately priced runtime buffers, KV/workspace allowance, and safety margin
from available memory.

`choose_capacity(budget)` returns `floor(usable_bytes / bytes_per_capacity_unit)`,
clamped to `[min_capacity, max_capacity]`:

- `max_capacity` is `index.num_experts`: at or above it, all experts stay
  resident and misses never happen (the zero-miss configuration).
- `min_capacity = max_router_fanout + staging_slots` (default `staging_slots=2`):
  a pool must hold at least every distinct expert one token activates in a layer,
  plus a little slack to load a miss while the active set is in use.
- If the budget cannot afford `min_capacity`, it **fails closed** with
  `StreamingCapacityError` (the host cannot serve this package within the
  budget) rather than picking an unworkable sub-fanout capacity.

`package_capacity_budget(...)` assembles the budget for a package. The
`resident_base_bytes` is measured from the package, so refusals scale with the
model. The generic allowances are environment knobs:

- `MOESPRESSO_SSD_KV_ALLOWANCE_GB` (default 1, Qwen4 default 2): reserved for
  KV cache and prefill workspace.
- `MOESPRESSO_SSD_SAFETY_MARGIN_GB` (default 2): headroom floor.

DeepSeek-V4 publishes enough attention geometry to price its composite cache
directly: 366,120,960 fixed bytes plus the manifest-derived per-token growth.
Unless `MOESPRESSO_SSD_KV_ALLOWANCE_GB` is explicit, the planner reserves that
exact amount for the served context. If the default 128K context cannot fit
beside the minimum expert pool, the runtime selects the largest safe
1K-aligned context and reports the reduction. An explicit
`--max-context-tokens` value is never reduced. Every resolved context below
128K emits a prominent usability warning, regardless of model family or whether
the lower limit was automatic or explicit.

Qwen4 reserves context-sized KVarN buffers in `runtime_resident_bytes` and adds
the configurable workspace allowance. Its larger default allowance leaves room
for transient prefill and checkpoint allocations before assigning expert slots.
SSD-backed PLE tables are excluded from fully resident model weights. Qwen4
does not apply DeepSeek's automatic context reduction.

### Memory budget at build time

`ssd_streaming_build._deterministic_available_bytes()` decides `available_bytes`:

- Automatic planning takes the smaller of `total RAM −
  MOESPRESSO_SSD_OS_RESERVE_GB` (default 5 GiB) and the reported wired-memory
  budget minus 1 GiB. A positive `iogpu.wired_limit_mb` supplies that budget;
  otherwise the planner uses Metal's recommended working-set size. The extra
  1 GiB provides conservative headroom for runtime and driver allocations.
  It is neither a measured model reservation nor evidence of a performance
  mechanism.
- The planner also subtracts the package's non-routed resident bytes, the
  KV/activation allowance and `MOESPRESSO_SSD_SAFETY_MARGIN_GB` (default 2 GiB).
  None is included in the 1 GiB envelope headroom.
- Qwen4 treats live available memory as an additional startup ceiling, even
  when it is close to the automatic wired-memory ceiling. Other automatic
  paths apply the live limit when it falls below 75% of that ceiling.
- If a family loader has hydrated the non-routed core, the planner adds its
  header-counted payload bytes back to the live reading. The capacity budget
  charges that payload separately, once.
- `MOESPRESSO_SSD_MAX_MEMORY_GB` (the `--max-memory-gb` CLI flag) sets an
  explicit startup planner ceiling in place of the wired-memory heuristic.
  Physical-memory reserve and live-pressure limits still apply. Experiments can
  exceed the automatic ceiling. The value selects routed-expert pool geometry;
  it does not limit RSS or resize the pool as context grows.
  The planner reserves the configured fixed KV/activation allowance before it
  assigns expert slots. Pool capacity and hit-rate behavior reproduce that
  operating point; miss *costs* remain optimistic on a larger host whose page
  cache can retain package data.

When `capacity_per_layer` is not supplied, the builder computes the budget,
calls `choose_capacity`, and records the result. The payload includes resolved planner
bytes, wired-budget source, automatic or explicit ceiling, and limiting source.
`capacity >= num_experts` uses the all-resident case of the same code path.

Temporary prefill allocations sit outside the startup budget. For long bounded
Qwen4 prompts, generation caps chunks at 512 tokens when active MLX memory
leaves less than 7 GiB under Metal's recommended working-set size. It also caps
MLX's free-buffer cache at 4 GiB while leaving 3 GiB of currently available
host memory outside it. An existing lower cache limit stays in force; the
previous limit returns after prefill. Chunk splits preserve disk-checkpoint
boundaries. The policy affects prefill memory and latency while expert capacity
and decode routing stay fixed. Other processes can still exhaust memory later;
this policy cannot recover from OOM.

---

## 5. Persistent slot pools (`expert_slot_pool.py`)

`ExpertSlotPool` is a persistent set of codec-native MLX buffers for one
(layer, projection). Its capacity is fixed between adaptive-growth
transactions. MXFP4 uses packed and scales,
K-quant uses wire arrays, and IQ_K allocates the streams exposed by
`IqkSwitchLinear`. A miss overwrites one slot and publishes residency only
after every declared component or stream has landed.

Bookkeeping is host-side integer state:

- `_slot_of: dict[expert -> slot]` and `_expert_at: list[slot -> expert]`.
- `_freq` (LFU counters) and `_recency` (a monotonic clock stamp per expert).
  Recency stamps replace an O(n) Python LRU list so a touch is O(1).

### Residency and eviction (decaying LFU)

Each layer has its own expert pool. Eviction selects its lowest-frequency
eligible expert, breaking ties by least-recent use. Other layers retain their
resident sets.

`ensure(expert_ids, *, protect, fence)` makes every requested expert resident in
three stages:

1. **Phase 1 (bookkeeping only):** count hits (touch their counters), and for each
   miss reserve a slot: a free slot if one exists, else evict a victim chosen by
   `_choose_slot`. Eviction order is LFU with a recency tie-break (smallest
   frequency, then oldest recency stamp); the `lru` policy uses recency alone.
   Reservations mark slot *occupancy* but do not yet publish residency.
2. **One fence:** if any reserved slot reclaimed a victim, a single
   `mx.synchronize()` is issued before any `pread` overwrites a slot, so in-flight
   GPU work that may still read old slot contents finishes first. The fence count
   is at most one per batch, regardless of the eviction count.
3. **Phase 3 (loads):** `pread` each missing expert's bytes into its slot, then
   publish `_slot_of[expert] = slot`. Residency is published only *after* the
   bytes land, so a failed `pread` never leaves the pool believing an expert is
   resident over stale bytes (fail-closed; unwound on error).

`protect` pins experts against eviction during a call (used by the prefill
chunk-ahead overlap so chunk `i`'s experts cannot be victims while chunk `i+1`
loads). The caller guarantees `|active ∪ protect| <= capacity`. If the active set
cannot fit, `ExpertCapacityExceeded` is raised; the SwitchGLU layer falls back to
chunking.

### Transactional adaptive growth

`MOESPRESSO_SSD_GROWTH_MAX_EXTRA_GB` lets the post-generation hook grow hot
routed layers using demand from a completed request. Its zero default keeps
startup capacity fixed. Growth never shrinks a pool. Each routed layer grows
its distinct gate, up, and down projection pools in one transaction:

1. Mark growth pending for the projection set. Demand loads and pool prefetch
   are storage writers; the transaction stops new
   writers and drain existing ones without holding sibling locks.
2. Allocate and evaluate complete replacement buffers without changing the
   live pools. The old and replacement allocations coexist during this step.
3. Fence prior readers, copy live rows and slot ledgers under the projection
   locks, then publish every replacement and reset the route certificates
   together. Allocation or copy failure before publication leaves that layer's
   live storage, mappings, and capacity unchanged.
4. Record the committed layer capacity, then seed newly available slots from
   observed hotness. Gate, up, and down rows land before any projection map is
   published. A seed failure restores the exact post-growth maps and
   reservations. A later layer or seed failure does not erase an earlier
   committed growth transaction.

Adaptive growth runs after generated cache state has been published. A hard
growth or seed failure is recorded in runtime statistics, logged once, and
latched off for later requests. Serving continues at the capacities that were
successfully committed.

The planner keeps two memory limits separate. The extra-capacity budget charges
only the long-lived slot delta beyond startup capacity. Replacement headroom
must also hold the complete target allocation, including spare slots, beside
the old layer while its transaction runs. The default live-memory floor leaves
4 GiB available. After a commit releases the old storage, only the net delta is
charged before the next layer is considered.

### One-pread misses via the shared row cache

Since the package stores one contiguous bundle row per (layer, expert), a
layer's three projection pools share a `BundleRowCache`. On a miss the first
pool to ask reads the whole row once into a staging buffer; the other two
consume it from the cache. Direct-copy codecs take their declared component
slices. IQ_K splits its `blocks` slice into the kernel-native streams for that
member. The cache is thread-safe, deduplicates concurrent row loads, and drops a
row after every projection consumer has taken it. A standalone pool falls back
to exact per-component reads through the same index.

`bundle_row_read_bytes` counts requests through this staging cache. OS page
cache hits can satisfy those reads without SSD traffic. Staging reuse also
changes the count while selected experts and model computation stay fixed.
Measure physical process reads separately for storage diagnosis.

### Cold-start hotlist seeding

`seed_hot(limit)` fills free slots with the highest-`_freq` experts not yet
resident. A cold pool's `_freq` is empty, so it is seeded from a hotlist before
serving when the package supplies a hotlist (see §8): the build installs prior
demand counts into the pools, seeds the
hottest experts into free slots, then rescales the installed prior so no entry
exceeds a cap: a high raw count must not make a seeded expert effectively
un-evictable and poison LFU adaptation against live traffic.

### In-session hotness decay

`MOESPRESSO_SSD_HOTNESS_DECAY_TOUCHES` (default 128; 0 disables) halves all LFU
counters every N touches inside `_touch`. With ~8 touches per token per pool at
top-8, 128 touches ≈ 16 tokens, so the effective frequency window is the recent
few hundred touches. The pool can follow topic shifts in long sessions, while
the recency tie-break stays fixed. Counters remain process-local. Serving does
not persist a learned demand hotlist across processes. Disk KV checkpoints
restore model state without these counters or the pool's resident expert set.

### On-device slot table

Each pool keeps an `mx.array` slot table of length `num_experts`, value = slot or
a `num_experts` sentinel for "absent". It is rebuilt lazily only when residency
changes (`_slot_table_dirty`), and crucially does **not** depend on the routing
`indices`, so it never forces the per-token routing graph. `remap_ondevice(indices)`
is then a pure on-device gather `slot_table[indices]` (no host round-trip),
numerically identical to the host `remap_loaded`. The caller must have `ensure`d
all active experts first so no sentinel reaches the kernel.
Pooled projections use the on-device gather after ensuring residency.

---

## 6. SwitchGLU forward over pooled experts (`pooled_switchglu.py`)

Each pooled projection is backed by an `ExpertSlotPool` and presents the
codec's native matmul contract over slot ids. The K-quant and MXFP4 projections
use their own wire layouts. `PooledIqkSwitchLinear` calls the `mlx_iqk` module whose
streams the pool owns. Replacing original expert ids with slots changes only
weight placement; the codec kernel sees the same encoded expert bytes.

`PooledSwitchGLU` owns the whole SwitchGLU seam (sort/gather, gate/up
activation, down, scatter) so a class-level `SwitchGLU` monkeypatch cannot
bypass it and a mixed-codec package keeps its declared behavior. Its bounded
path:

- Reads the router indices to the host once, counts active experts, and
  classifies the call as decode or prefill. Full-resident K-quant and IQ_K
  routes consume router ids on device and skip this synchronization.
- If the active set exceeds capacity, chunks the operation. The exact sorted
  crossover and projection kernel are codec-specific.
- The direct path ensures all three projection pools (in parallel on a 3-worker
  executor when two or more pools have misses), then runs the projections.

The generic IQ_K route retains the `mlx_iqk` execution policy. Below 4,096 routed
pairs it uses the gate, up, and down GEMV modules. At or above that crossover it
sorts once and runs range-dequantized projections with a default split count of
16. Full expert capacity with identity slots selects a host-sync-free route.
Smaller capacities use the same math after loading, eviction, slot remapping, and
chunking.

For Qwen4's 2560-wide hidden state and 640-wide routed experts, full-resident
IQ2_K, IQ2_KS and IQ3_K prefill uses packed matrix tiles. Gate and up share an
activation tile and produce FP16 SwiGLU activations. Down reads the stored
768-column layout and multiplies the 640 non-padding columns. Weights are
reconstructed in threadgroup memory, with no expert-sized decoded weight array.
The route requires batch size one, matching gate/up codecs and identity slot
maps. It keeps the existing GEMV-to-FP16-matrix crossover: smaller chunks,
ordinary decode and unsupported layouts use their existing paths.
`iqk_packed_prefill_calls` and `iqk_packed_prefill_pairs` track engagement
separately from the route-classification counters.

## 7. Routed decode scheduling

The pool never splits one routed layer into resident and missing partial
matmuls. Misses land before compute, then one `PooledSwitchGLU` operation runs
the routed layer. Metal dispatch geometry is codec-specific. MXFP4 has a
single-dispatch routed kernel, K-quant uses its fused routed kernels, and IQ_K
retains the `mlx_iqk` gate, up, and down kernel family. Python sees one routed
operation in every case.

### Barrier-free full-resident decode

Package loading installs one `PooledDecodeSession` across all routed layers
for DeepSeek, Ornith and Qwen4. `pooled_moe.run_pooled_moe` handles demand
loading, publication and request drainage at full and bounded capacity. Model
adapters preserve their router and reduction math. Qwen4 also preserves
independent projection-slot maps, the padded down projection and source-expert
reduction order. A capacity certificate lets full residency skip loading work
within the same model graph.

The request-owned scheduler handles execution. Native-gate availability
determines its synchronization mechanism.

The bounded pipeline overlaps expert-miss service with compute. When every
projection pool holds the whole expert set, the residency certificate selects a
lighter route: blocks skip ring export, the event gate, worker submission, and
per-layer demand kicks. Router ids stay on device. K-quant keeps its combined
gate/up and down route; IQ_K calls the stream modules owned by each pool directly
and uses its measured decode/verify commit cadence. Slot remapping preserves
encoded expert rows. Compare full and bounded outputs with the same routing
policy: Qwen4's default bounded preference can select different experts from
its unbiased full-resident path.

Any partial-residency session fails the certificate closed and keeps the
pipeline, where there are misses to overlap.

Supported source-MXFP4 geometry uses a fused single-dispatch routed kernel.
K-quant and IQ_K retain their codec-specific dispatches and eligibility checks.

### The decode driver: ring export + worker + commit ordering

`PooledSparseMoeBlock` is the MoE block. On a decode token (`MOESPRESSO_SSD_RING_DECODE`,
default on, after a one-time GPU->host visibility self-test passes):

1. Compute router gates and select the top-k experts on device.
2. A tiny export kernel writes the selected ids + a monotonic sequence number into
   a persistent per-layer ring buffer (relaxed device-scope atomics, guarded by a
   seqlock and an FNV checksum against torn/stale reads).
3. Build the routed island graph against the persistent per-layer slot-id buffers
   (no host values needed yet) and add the resident shared expert.
4. Kick the block output (`async_eval`) so the GPU runs this layer's routed work
   while Python builds the next layer's graph (the block-exit kick).
5. Submit a single **ordered** worker (`max_workers=1`, FIFO == layer order) that
   seqlock-polls the ring from raw memory (zero MLX), reads the ids, runs
   `ensure()` for the misses, and publishes the slot ids into the layer's buffers
   in place, `ring_install`. The worker does no MLX encoding (MLX command
   encoders are thread-local; commits stay on the main thread).

The ordering invariant is that a layer's slot publication precedes the commit of
its routed graph. In the base ring path that is enforced by committing the
previous layer's routed graph only after its worker future resolves; the deepest
MoE layer (`pipeline_is_last`) drains the queue at the end of the token. The ring
watchdog raises `TimeoutError` and prevents stale routing if
the GPU export never becomes host-visible.

`begin_projection_load` / load tickets implement the non-ring overlap path: once
router indices are known, start the routed loads, force the resident shared expert
while the reads are in flight, then wait only on the unresolved tail before the
routed matmul.

Cross-chunk predictive prefetch extends the overlap across prompt chunks on the
over-capacity sorted-chunked prefill path, where the per-call overlap seam
declines. After a layer's over-capacity call finishes for one prompt chunk, the
layer submits a background best-effort prefetch of the experts that call used and
stores it as a ticket; the layer's next call (the next prompt chunk) awaits the
ticket before its chunk-ahead path runs, so slots that would miss are already
warm. Per-layer pools make the prefetch the only pool mutator between the two
calls, and the pool's `prefetch` primitive protects the last demand set, so the
final chunk's still-executing slots are never evicted. A mismatched or stale
ticket is counted and discarded; the normal per-call sync and ensure service any
difference, so the prediction only moves bytes, never routing. Measured at
cap-192: 37K prefill 516.7 to 560.8 t/s and 4K prefill 620.2 to 677.4 t/s, with
the miss volume down about 30 percent, token-identical across capacities. The
full-capacity build never dispatches over capacity, so the certificate path never
submits or consumes a ticket. This evidence covers K-quant. IQ_K disables the
cross-call prediction because its one-chunk DS4 prefill has no later prefill
consumer and each speculative landing also performs row-to-stream relayout.
Engagement counters
(`prefetch_ticket_submitted/consumed/mismatched/stale/experts/loaded`) are
exported in the streaming stats.

### Optional native gate (`native_gate.py`)

When the native MTLSharedEvent gate extension is built and passes its self-test,
the decode path (`MOESPRESSO_SSD_GATE_DECODE`, default on) puts each layer's routed
island behind an in-stream event wait: the main thread commits the whole layer
immediately with no per-layer join, and the worker signals the event after
`ensure` + publish. Kernels wait for IO; threads never wait for kernels. The gate
is always signaled (even on worker error, poison) so a stuck GPU wait cannot
outlive a token; errors surface at the once-per-token future drain.

The gate is **optional**. `load_gate()` uses the installed `moespresso._native`
extension or an explicit `MOESPRESSO_NATIVE_DIR` override. Decode uses the ring
path if the extension is absent, fails to import, or fails its once-per-process
hold, foreign-thread release and value-integrity self-test.
`MOESPRESSO_SSD_GATE_DECODE=0` disables it. The ring path itself similarly falls
back to a legacy path if the ring visibility self-test fails.

Qwen4 uses native slot publication automatically when all selected experts are
resident and the runtime is eligible. It requires
`qwen-native-all-hit-gate-v2` and three independent IQ_K relayout pools, without
spare slots or active prefetching. Under projection locks, the shared worker
checks the actual GPU-exported route, publishes all three slot maps and signals
the same event before reacquiring the Python GIL. The normal reader continues
demand accounting without rewriting a successfully published slot buffer.

The native attempt pins buffers once and releases the GIL while polling. Its
first slice yields without delay; later slices use bounded exponential sleep
backoff. Each checks readiness before yielding and polls for at most 1 ms. The
worker releases projection locks and checks cancellation between slices.
Incomplete exports remain eligible until ready or until the overall ring timeout.
Pool ownership and membership stay fixed during the wait. Misses use the normal
reader; full residency keeps its no-sync path. Missing native capability or
unsupported pool geometry uses the shared reader without early publication.

Health counts native polling calls in `qwen_native_publication_poll_slices` and
slices ending before route readiness in `qwen_native_publication_poll_yields`.
Such yields are normal under load. `qwen_native_publication_pending` and
`qwen_native_publication_timed_out` count terminal deadline failures; queued
work and successful delayed publications do not increment them.

---

## 8. Cold-start residency seeding (`ssd_streaming_build.py`)

At build time, after the pools are installed and the non-routed core is resident,
`seed_expert_residency` warms the pools so the first tokens do not pay a full cold
miss. Startup follows this order:

1. **Explicit full prewarm** (`MOESPRESSO_SSD_PREWARM_EXPERTS=all`): load every
   expert now; fails closed when any pool cannot hold the full expert set.
2. **Default full prewarm at full capacity**: when every projection pool's
   capacity covers the full expert set and no explicit prewarm is requested,
   the build prewarms all experts (source `all-default`). Pool residency
   selects the routed prefill kernel, and a cold pool at full capacity would
   serve the segmented numerics until the pools fill, diverging from the
   gate-certified barrier-free path at knife-edge tokens on long prompts.
   Prewarming at load pins serving to the gate-certified numerics and moves
   the cold first-request SSD reads into load time, which measured faster than
   package-hotlist seeding of the same expert set.
   Partial-capacity configurations are unaffected.
3. The **package's fixed hotlist**, when supplied. The builder records its
   provenance. The hotlist contains no saved record of a previous user's
   requests. Without one, demand fills the pools.

`load_expert_hotlist` installs package demand counts into all three pools per
layer and seeds their hottest experts into free slots during build. It caps the
prior at `prior_cap` (default 8); raw imatrix counts can reach hundreds of
thousands and would impede eviction. Seeding uses only free slots, preserving
live demand.

The three projection consumers are seeded in shared row-cache windows. Gate,
up, and down therefore consume one bundle row while it is live instead of
reading the row once per projection when the seed capacity exceeds the cache
window.

`expert_locality.py` is the offline analysis that justifies this: given a trace of
per-layer selected experts, it computes the activation histogram
(`summarize_trace`), the per-layer hotlist seed (`hotlist_from_counts`), and the
simulated per-layer LRU hit rate for a given cache size, optionally pre-warmed by
a seed (`simulate_lru_hit_rate`, `coverage_curve`). It is pure and import-light
(no MLX), so it is fully unit-tested without running a model. Experts are cached
per `(layer, expert)`: expert 5 in layer 3 is a different weight than in layer 7.

---

## 9. Single-owner run lock (`streaming_run_lock.py`)

A streaming run holds a large working set within a tight memory budget; two
concurrent runs can exceed unified-memory headroom even when each is safe alone.
`build_ssd_streaming_model` acquires a non-blocking process lock
(`flock(LOCK_EX | LOCK_NB)` on `/tmp/moespresso-ssd-streaming.lock`) and holds it
for the model's lifetime (the lock object is attached to the built model). If
another process owns it, `SSDStreamingAlreadyRunning` is raised with a clear
message. `MOESPRESSO_ALLOW_PARALLEL_SSD_STREAMING=1` overrides the guard;
`MOESPRESSO_SSD_STREAMING_LOCK_PATH` overrides the lock path.

---

## 10. Environment knobs

| Variable | Default | Effect |
| --- | --- | --- |
| `MOESPRESSO_SSD_MAX_MEMORY_GB` (`--max-memory-gb`) | unset | Explicit startup capacity-planner ceiling in place of the automatic wired-budget heuristic. RSS is measured separately. |
| `MOESPRESSO_SSD_OS_RESERVE_GB` | 5 | RAM held back from the deterministic budget. |
| `MOESPRESSO_SSD_KV_ALLOWANCE_GB` | unset | Fixed KV-cache / workspace allowance. DeepSeek-V4 derives a context-sized reservation when unset; Qwen4 reserves KVarN separately and adds this allowance (default 2 GiB). Ornith uses the fixed allowance. |
| `MOESPRESSO_SSD_SAFETY_MARGIN_GB` | 2 | Safety headroom in the capacity budget. |
| `MOESPRESSO_SSD_PREWARM_EXPERTS` | unset | `all` forces a full expert prewarm at load (fails closed below full capacity); `none` skips explicit and default full prewarm, then uses the package hotlist. |
| `MOESPRESSO_SSD_HOTNESS_DECAY_TOUCHES` | 128 | Halve LFU counters every N touches (`0` disables). |
| `MOESPRESSO_SSD_GROWTH_MAX_EXTRA_GB` | 0 | Cumulative adaptive-growth allowance beyond startup capacity. The zero default preserves the startup pool budget. Detached replacement storage must separately fit above the live-memory floor. |
| `MOESPRESSO_SSD_ROUTE_TRACE_HIDDEN` | 0 | Diagnostic: also capture decode router-input hidden states in route-trace study runs (`1` enables; large captures). |
| `MOESPRESSO_SSD_RING_DECODE` | on | GPU ring-export decode driver (`0` = legacy path). |
| `MOESPRESSO_SSD_RING_TIMEOUT` | 10.0 | Worker ring-poll timeout (seconds). |
| `MOESPRESSO_SSD_GATE_DECODE` | 1 | Use the native MTLSharedEvent gate if built (`0` disables). |
| `MOESPRESSO_NATIVE_DIR` | unset | Search dir for the native gate extension. |
| `MOESPRESSO_ROUTED_DECODE_SPLIT` | 2 | Output-split factor for the routed decode kernel. |
| `MOESPRESSO_ALLOW_PARALLEL_SSD_STREAMING` | unset | Bypass the single-owner run lock. |
| `MOESPRESSO_SSD_STREAMING_LOCK_PATH` | `/tmp/moespresso-ssd-streaming.lock` | Run-lock path. |
