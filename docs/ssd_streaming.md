# SSD-streaming MoE runtime

The SSD-streaming runtime serves a routed Mixture-of-Experts model whose routed
expert weights do not fit in memory. Only the non-routed core (attention, norms,
router gates, shared experts) stays resident; the routed experts live on disk and
are streamed in on demand, into a fixed per-layer pool of MLX buffers, so a large
MoE serves within a bounded memory budget.

The design follows a hard rule: there is no Python/numpy expert *tensor* compute
on the per-token path. The host is allowed only small integer bookkeeping (read
the router's selected expert ids, look up which slots they occupy, decide which
to load) plus the byte-range `pread` of a missing expert. All expert math runs in
MLX/Metal kernels, and the routed MLP is dispatched as one fused operation per
layer.

Source: `src/moespresso/runtime/`.

---

## 1. Component map

| File | Role |
| --- | --- |
| `ssd_streaming_build.py` | Build entry point: take the run lock, build the index, compute capacity, install pooled SwitchGLUs, load the non-routed core resident, seed cold-start residency. |
| `expert_index.py` | Per-layer bundle byte-offset index; maps `(layer, expert[, projection, component])` to an exact byte range in a shard. |
| `expert_loader.py` | Single-expert byte-range loader (`pread` one expert into a fresh `mx.array`); proof/test helper that uses the same primitive the pools use. |
| `pread_into.py` | The direct-IO primitive: `os.preadv` a file byte range straight into a writable MLX buffer (no Python `bytes`), with a bounded fd cache. |
| `expert_slot_pool.py` | The persistent fixed-capacity slot pool per (layer, projection); LFU residency, miss loads, hotness decay, page-cache hygiene, on-device slot table. |
| `expert_pool.py` | Compact-pool proof primitive that validates bit identity between a remapped compact `packed` tensor and the full stack. The product hot path uses the persistent slot pools. |
| `expert_locality.py` | Pure router-locality analysis: activation histogram, per-layer hotlist, simulated LRU hit-rate curve. |
| `streaming_capacity.py` | Capacity math: byte cost per capacity unit from the index, memory budget, `choose_capacity`. |
| `pooled_switchglu.py` | The SwitchGLU forward over pooled experts and the single-dispatch routed decode driver (ring export + worker + compiled island). |
| `gather_tq_split_norms.py` | MoEspresso-owned fork of jang's gather-TQ kernel that decouples `norms` (full-resident) from `packed` (streamed pool). |
| `routed_decode_kernel.py` | MoEspresso-owned single-dispatch Metal kernel that fuses the whole routed MLP for one decode token. |
| `native_gate.py` | Loader for the optional native MTLSharedEvent gate extension; falls back transparently if absent. |
| `streaming_run_lock.py` | Single-owner process lock so only one real-model streaming run executes at a time. |

---

## 2. The expert byte index (`expert_index.py`)

A package stores one routed layer's experts as a single **bundle** tensor
`...switch_mlp.experts.tq_bundle`, a `uint8 [n_experts, row_bytes]` array whose
row `e` concatenates expert `e`'s full payload (gate/up/down `packed` + `norms`)
in a fixed `ROW_ORDER`. The within-row geometry (each component's offset, byte
count, shape, dtype, and the projection bit-width) travels in the shard's
safetensors `__metadata__`.

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
components tile each row in `ROW_ORDER` with no padding, first/last rows stay in
bounds) and returns a list of problems.

Pre-bundle "stacked" packages (separate `tq_packed`/`tq_norms`/`tq_bits` tensors)
are **not readable**: the index raises `StackedLayoutError` with a re-convert
message rather than silently missing an expert. A package that mixes bundle and
stacked tensors, or has inconsistent `num_experts`, also fails loud.

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

`expert_loader.load_expert(...)` is the single-expert form: allocate a fresh
`mx.array` of the right shape/dtype from the index and `pread` the bytes in. The
bytes stay packed (`uint32`); no TQ dequant happens at load time. Jang's kernel
runs the packed weights. This module is a proof/test helper; the product miss
path uses the same `pread_view_cached` primitive to fill persistent pool slots in
place.

---

## 4. Capacity math (`streaming_capacity.py`)

Capacity is derived from the memory contract. One **capacity unit** means
"one resident expert slot in every routed (layer, projection) pool", so its byte
cost is summed across the whole model.

- `bytes_per_layer_slot(index)`: for each routed layer, the `packed`+`norms`
  bytes of one expert summed over its projections, read from the index geometry.
- `bytes_per_capacity_unit(index)` = the sum of those over all layers: the
  marginal RAM cost of raising capacity by one slot everywhere.
- `non_routed_payload_bytes(package_dir)`: header-only sum of every non-bundle
  payload tensor; this is the resident base the runtime keeps at startup.

`CapacityBudget` carries `available_bytes`, `resident_base_bytes`,
`kv_activation_allowance_bytes`, `safety_margin_bytes`, `bytes_per_capacity_unit`,
`min_capacity`, `max_capacity`. Its `usable_bytes` subtracts the base, KV
allowance, and safety margin from available memory.

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
model. Two allowances are environment knobs:

- `MOESPRESSO_SSD_KV_ALLOWANCE_GB` (default 1): reserved for KV cache /
  activations.
- `MOESPRESSO_SSD_SAFETY_MARGIN_GB` (default 2): headroom floor.

### Memory budget at build time

`ssd_streaming_build._deterministic_available_bytes()` decides `available_bytes`:

- The base is `total RAM − MOESPRESSO_SSD_OS_RESERVE_GB` (default 5 GiB). Using
  total-minus-reserve rather than instantaneous available memory makes the chosen
  capacity deterministic run-to-run on an idle host (macOS reports reclaimable
  page cache inconsistently in `available`). When live `available` is within 25%
  of the deterministic number, the gap is treated as reclaimable cache and the
  deterministic number is trusted; under genuine memory pressure the budget
  clamps to live `available`.
- `MOESPRESSO_SSD_MAX_MEMORY_GB` (the `--max-memory-gb` CLI flag) caps the
  startup capacity-planner input. It selects a smaller routed-expert pool, but
  it is not an RSS limit and does not resize the pool as the context grows.
  The planner reserves the configured fixed KV/activation allowance before it
  assigns expert slots. Pool capacity and hit-rate behavior reproduce that
  operating point; miss *costs* remain optimistic on a larger host whose page
  cache can retain package data.

When `capacity_per_layer` is not supplied, the builder computes the budget,
`choose_capacity`s it, and records the budget payload. `capacity >= num_experts`
yields the all-resident case of the same code path.

---

## 5. Persistent slot pools (`expert_slot_pool.py`)

`ExpertSlotPool` is a fixed-capacity, persistent set of MLX buffers for one
(layer, projection). The pool arrays (`packed (capacity, out_features,
packed_cols) uint32` and `norms (capacity, out_features) float16`) are allocated
once and reused; a miss overwrites a slot **in place** via `pread`.

Bookkeeping is host-side integer state:

- `_slot_of: dict[expert -> slot]` and `_expert_at: list[slot -> expert]`.
- `_freq` (LFU counters) and `_recency` (a monotonic clock stamp per expert).
  Recency stamps replace an O(n) Python LRU list so a touch is O(1).

### Residency and eviction (LFU-style)

`ensure(expert_ids, *, protect, fence)` makes every requested expert resident in
two phases under a bookkeeping lock:

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

`grow(capacity)` enlarges a pool in place (copying live and spare rows into new,
larger buffers) without shrinking.

### One-pread misses via the shared row cache

Since the package stores one contiguous bundle row per (layer, expert), a layer's
three projection pools share a `BundleRowCache`. On a miss the first pool to ask
`pread`s the whole row once into a staging buffer; the other two consume it from
the cache and `memcpy` their `packed`/`norms` slices out of it. The cache is
thread-safe (the three pools may load in parallel on the projection executor; an
in-flight marker dedups concurrent loads of the same row) and drops a row after
`consumers` takes, so the steady state is an empty cache. A pool built standalone
(no cache, as in tests) falls back to exact per-component `pread`s through the
same index.

### Cold-start hotlist seeding

`seed_hot(limit)` fills free slots with the highest-`_freq` experts not yet
resident. A cold pool's `_freq` is empty, so it is seeded from a hotlist before
serving (see §8): the build installs prior demand counts into the pools, seeds the
hottest experts into free slots, then rescales the installed prior so no entry
exceeds a cap: a high raw count must not make a seeded expert effectively
un-evictable and poison LFU adaptation against live traffic.

### In-session hotness decay

`MOESPRESSO_SSD_HOTNESS_DECAY_TOUCHES` (default 128; 0 disables) halves all LFU
counters every N touches inside `_touch`. With ~8 touches per token per pool at
top-8, 128 touches ≈ 16 tokens, so the effective frequency window is the recent
few hundred touches. This lets the pool follow topic shifts in long sessions
instead of letting early-session experts squat on inflated counts; the recency
tie-break is unchanged, and a persisted hotlist then reflects recent demand.

### Page-cache hygiene (advisory only)

`MOESPRESSO_SSD_EVICT_DONTNEED` (default 1; `=0` disables) advises the kernel to
drop an evicted expert's file pages after a demand eviction. `_advise_dontneed`
maps the evicted row read-only, calls `madvise(MADV_DONTNEED)`, and unmaps. It is
**advisory-only on a read-only mapping. It cannot corrupt data and never
raises**; the worst case is an ignored hint or a slower re-read. Errors are
counted and suppressed. The advice is issued only from the gate-projection pool
(one bundle row covers all three projections) and only on demand evictions, with
the advisory syscalls performed outside the bookkeeping lock.

### On-device slot table

Each pool keeps an `mx.array` slot table of length `num_experts`, value = slot or
a `num_experts` sentinel for "absent". It is rebuilt lazily only when residency
changes (`_slot_table_dirty`), and crucially does **not** depend on the routing
`indices`, so it never forces the per-token routing graph. `remap_ondevice(indices)`
is then a pure on-device gather `slot_table[indices]` (no host round-trip),
numerically identical to the host `remap_loaded`. The caller must have `ensure`d
all active experts first so no sentinel reaches the kernel.
`MOESPRESSO_SSD_ONDEVICE_REMAP` (default on) selects this over the host remap.

---

## 6. SwitchGLU forward over pooled experts (`pooled_switchglu.py`)

`PooledTurboQuantSwitchLinear` is one routed TQ projection backed by an
`ExpertSlotPool`. `matmul_slots(x, remapped_indices)` calls jang's
`gather_tq_matmul` over `pool.packed`/`pool.norms` with indices already remapped
to pool slots. The kernel reads `n_experts` from `packed.shape`, so a compact pool
of `capacity` slots is bit-identical to a full stack. That equivalence is what
`expert_pool.py` validates as the foundational primitive.

`PooledSwitchGLU` owns the whole SwitchGLU seam (sort/gather, gate/up activation,
down, scatter) so jang's class-level `SwitchGLU` monkeypatch cannot bypass it, and
so a mixed-bit gate/up package is handled correctly. Its `__call__`:

- Reads the router indices to the host once (small integer array), counts unique
  active experts, classifies the call as decode (one token-layer) or prefill.
- If the active set exceeds capacity, falls back to chunking: sorted chunks
  (`_call_sorted_chunked`) when `indices.size >= 64` (prefill), else row chunks
  (`_call_chunked`). Otherwise it goes through the direct path.
- The direct path ensures all three projection pools (in parallel on a 3-worker
  executor when two or more pools have misses), then runs the projections.

**Fused gate+up.** When gate and up share codebook, signs, and bits (true for real
packages), `MOESPRESSO_SSD_FUSED_GATE_UP` (default on) computes `SiLU(gate)*up` in
one Metal dispatch via jang's `fused_gate_up_swiglu_matmul`, then runs down on the
gather path. The precondition is checked at construction; anything else falls back
to the exact separate path (two gather kernels plus a Python activation),
preferring correctness over speed.

### Norms decoupled from slots (`gather_tq_split_norms.py`)

This is a MoEspresso-owned fork of jang's gather-TQ kernel. jang indexes both
`packed` and `norms` by the same remapped slot, which would force `norms` to be
re-`pread` and slotted on every miss. The fork adds a separate `norms_indices`
input so `norms` can be a **full-resident** array indexed by original expert id
while `packed` stays the small streaming pool indexed by slot. The norms then
never miss or stream. It is byte-faithful to jang's kernel except for the two
norms lines, and with `norms_indices == rhs_indices` it is identical to the
upstream kernel (pinned by test).

---

## 7. Single-dispatch routed decode

For decode (one token per layer) the routed island's cost is dependent-chain
*latency*: rotate -> fused gate/up -> rotate -> down would be four serial Metal
dispatches per layer, each paying launch + drain. The fix is to fuse the whole
routed MLP into **one dispatch per layer**, with one dispatch boundary, and never
to split a layer into resident-plus-missing partial matmuls. Misses are resolved
into the pool *before* the single fused operation runs, so the matmul always sees
one residency state.

### Barrier-free full-resident decode

The pipeline above overlaps expert-miss service with compute; when there are no
misses to hide it pays its machinery cost for nothing. At full residency (every
projection pool holds the whole expert set, so the residency certificate holds),
the decode blocks skip the ring export, the event gate, the worker submit, and the
per-layer block-exit kick, and queue the whole token graph lazily, committing every
`MOESPRESSO_DSV4_DECODE_FLUSH_LAYERS` layers. The routed math is the same combined
gate/up gather, activation, and down gather the pipelined builder emits, so the
route is bit-identical to the ring path; only the index source (router ids consumed
on device rather than worker-published slot buffers) and the scheduling change.
Without this route the streamed forward builds one lazy MLX graph per routed layer
and flushes forty times per decode token, where the resident runtime builds one
graph and flushes once; the barrier-free route matches the resident shape and
recovers the full-capacity streamed decode rate. Each block gates the route on
the shared residency certificate plus its own kill switch: the DS4 block on
`MOESPRESSO_SSD_BARRIER_FREE_DECODE` (default on; `0` restores the pipelined
path), the Qwen block on `MOESPRESSO_SSD_DECODE_SCHED` (default on; `0`
restores the pipelined path). Any partial-residency session fails the
certificate closed and keeps the pipeline, which is the correct path when
there are misses to overlap.

Two mechanisms implement the single fused operation:

- **Compiled island.** `_get_compiled_island(K)` builds, once per `K`, an
  `mx.compile`d closure: on-device slot-table gather (x2) -> rotate -> fused
  gate/up/SwiGLU -> rotate -> down gather, following jang's decode-patch shape.
  Pool buffers, slot tables, and indices are runtime inputs, so residency changes
  outside the island never invalidate the trace. Slot-bank mutation
  (`ensure`/`pread`, slot-table rebuilds) stays strictly *outside* the island.
  `MOESPRESSO_SSD_COMPILED_ISLAND` (default on) gates it.
- **Routed decode kernel** (`routed_decode_kernel.py`). A MoEspresso-owned single
  Metal dispatch that computes, for one decode token and K experts: sign +
  Hadamard rotate of `x` (threadgroup memory), TQ-dequant gate/up matmul +
  SwiGLU + fp16 bottleneck, sign + Hadamard rotate of the activation, then the
  TQ-dequant down matmul. The grid is one threadgroup per (expert, output split),
  `MOESPRESSO_ROUTED_DECODE_SPLIT` (default 2) widening occupancy. The packed
  layout, norms semantics, rotation, and codebook dequant are byte-faithful to
  jang's kernels; only the reduction order differs, so outputs are numerically
  equivalent within a tight test tolerance. Bit identity is not required. Shapes
  are gated by `routed_decode_supported` (power-of-two dims, rotation fits
  threadgroup memory).

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
watchdog raises a loud `TimeoutError` rather than ever serving stale routing if
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
submits or consumes a ticket. Default ON; `MOESPRESSO_SSD_PREFETCH=0` is the kill
switch. Engagement counters (`prefetch_ticket_submitted/consumed/mismatched/
stale/experts/loaded`) are exported in the streaming stats.

### Opt-in decode lookahead (`MOESPRESSO_SSD_LOOKAHEAD`)

`MOESPRESSO_SSD_LOOKAHEAD=<delta>` (user-settable, default 0, off) engages
speculative expert loads during decode on the native-gate path: as each layer's
routed ids are exported, a bias-aware prediction targets the layer `delta`
positions ahead, and predicted experts load into spare slots carved out of the
same per-layer capacity budget (at most 16, so the spares displace LFU demand
slots rather than growing memory). Prediction only moves bytes; every layer's
own ensure still services any miss, so routing and output are unchanged, and
engagement is visible in the `lookahead_*` streaming stats. The build refuses
to carve spares when the native-gate decode path is not live or the pools are
at full residency.

Parked default-off: on a served A/B at the DS4 cap-48 anchor, the on arms cut
decode-phase misses about 15 percent but converted only +0.124 tok/s against a
+0.5 tok/s bar, with token streams bit-identical on both sides. Emulated
memory budgets under-price miss service through the page cache, so the default
stays off until the knob is re-priced under budgets backed by matching
physical memory.

### Optional native gate (`native_gate.py`)

When the native MTLSharedEvent gate extension is built and passes its self-test,
the decode path (`MOESPRESSO_SSD_GATE_DECODE`, default on) puts each layer's routed
island behind an in-stream event wait: the main thread commits the whole layer
immediately with no per-layer join, and the worker signals the event after
`ensure` + publish. Kernels wait for IO; threads never wait for kernels. The gate
is always signaled (even on worker error, poison) so a stuck GPU wait cannot
outlive a token; errors surface at the once-per-token future drain.

The gate is **optional**. `load_gate()` looks for the compiled extension (under
`native/gate/build` or `MOESPRESSO_NATIVE_DIR`); if it is missing, fails to
import, or fails its once-per-process self-test (proving hold + foreign-thread
release + value integrity), decode falls back transparently to the ring path.
`MOESPRESSO_SSD_GATE_DECODE=0` disables it. The ring path itself similarly falls
back to a legacy path if the ring visibility self-test fails.

---

## 8. Cold-start residency seeding (`ssd_streaming_build.py`)

At build time, after the pools are installed and the non-routed core is resident,
`seed_expert_residency` warms the pools so the first tokens do not pay a full cold
miss. Sources, in precedence order:

1. **Explicit full prewarm** (`MOESPRESSO_SSD_PREWARM_EXPERTS=all`): load every
   expert now; fails closed when any pool cannot hold the full expert set.
2. **Default full prewarm at full capacity**: when every projection pool's
   capacity covers the full expert set and no explicit prewarm is requested,
   the build prewarms all experts (source `all-default`). Pool residency
   selects the routed prefill kernel, and a cold pool at full capacity would
   serve the segmented numerics until the pools fill, diverging from the
   gate-certified barrier-free path at knife-edge tokens on long prompts.
   Prewarming at load pins serving to the gate-certified numerics and moves
   the cold first-request SSD reads into load time (~14 s on the byte-faithful
   DS4 package, and faster than saved-hotlist seeding of the same expert set).
   `MOESPRESSO_SSD_PREWARM_DEFAULT=0` restores lazy hotlist seeding.
   Partial-capacity configurations are unaffected.
3. **Saved demand** from a prior session (`save_expert_hotlist` persists each
   layer's gate-pool touch frequencies to a per-package path under the user cache,
   keyed by the manifest's content-addressed id).
4. The **package's imatrix-derived hotlist** shipped with the package.

`load_expert_hotlist` installs the demand counts into all three pools of each
layer, seeds the hottest experts into free slots now (moving cold misses into
build time), and rescales the installed prior so no entry exceeds `prior_cap`
(default 8): imatrix counts run to the hundreds of thousands and a raw install
would make seeded experts un-evictable. Seeding only fills *free* slots, so it can
never evict live demand. `MOESPRESSO_SSD_HOTLIST=0` disables the hotlist tiers;
combined with `MOESPRESSO_SSD_PREWARM_DEFAULT=0` it yields a fully cold pool
state (the diagnostic configuration).

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
| `MOESPRESSO_SSD_MAX_MEMORY_GB` (`--max-memory-gb`) | unset | Startup capacity-planner ceiling used to select expert-pool geometry; RSS is measured separately. |
| `MOESPRESSO_SSD_OS_RESERVE_GB` | 5 | RAM held back from the deterministic budget. |
| `MOESPRESSO_SSD_KV_ALLOWANCE_GB` | 1 | KV-cache / activation allowance in the capacity budget. |
| `MOESPRESSO_SSD_SAFETY_MARGIN_GB` | 2 | Safety headroom in the capacity budget. |
| `MOESPRESSO_SSD_HOTLIST` | 1 | Cold-start residency seeding (`0` disables). |
| `MOESPRESSO_SSD_PREWARM_EXPERTS` | unset | `all` forces a full expert prewarm at load (fails closed below full capacity); `none` skips both the prewarm and hotlist seeding. |
| `MOESPRESSO_SSD_PREWARM_DEFAULT` | 1 | Default full prewarm when every pool covers the full expert set (`0` restores lazy hotlist seeding). |
| `MOESPRESSO_HOTLIST_DIR` | user cache | Directory for saved-demand hotlists (tests, multi-user). |
| `MOESPRESSO_SSD_HOTNESS_DECAY_TOUCHES` | 128 | Halve LFU counters every N touches (`0` disables). |
| `MOESPRESSO_SSD_EVICT_DONTNEED` | 1 | Advisory page-cache drop on demand eviction. |
| `MOESPRESSO_SSD_GROWTH_MAX_EXTRA_GB` | 2 | Cap on adaptive pool growth beyond build-time capacity (`0` disables growth). |
| `MOESPRESSO_SSD_ONDEVICE_REMAP` | on | On-device slot-table gather vs host remap. |
| `MOESPRESSO_SSD_FUSED_GATE_UP` | on | One-dispatch fused gate/up vs separate kernels. |
| `MOESPRESSO_SSD_FUSED_SORTED_SWIGLU` | on | Fused sorted K-quant gate/up + SwiGLU kernel on the sorted routes (`0` restores the unfused pair). |
| `MOESPRESSO_SSD_UNIFIED_PREFILL` | on | Partial-residency sorted-chunked prefill through the same fused sorted kernels as the full-resident route (`0` restores the segmented chunked compute). |
| `MOESPRESSO_SSD_COMPILED_ISLAND` | on | `mx.compile`d single-dispatch routed island. |
| `MOESPRESSO_SSD_PREFETCH` | on | Cross-chunk predictive expert prefetch on the over-capacity streamed prefill path (`0` disables). |
| `MOESPRESSO_SSD_LOOKAHEAD` | 0 | Decode lookahead depth in layers; a positive value loads predicted experts into spare slots on the native-gate decode path (parked off; see the lookahead subsection). |
| `MOESPRESSO_SSD_BARRIER_FREE_DECODE` | on | Barrier-free full-resident decode for the DS4 block (`0` restores the pipelined ring/gate decode). |
| `MOESPRESSO_SSD_DECODE_SCHED` | on | Barrier-free full-resident decode scheduling for the Qwen block (`0` restores the pipelined ring/gate decode). |
| `MOESPRESSO_SSD_ROUTE_TRACE_HIDDEN` | 0 | Diagnostic: also capture decode router-input hidden states in route-trace study runs (`1` enables; large captures). |
| `MOESPRESSO_SSD_RING_DECODE` | on | GPU ring-export decode driver (`0` = legacy path). |
| `MOESPRESSO_SSD_RING_TIMEOUT` | 10.0 | Worker ring-poll timeout (seconds). |
| `MOESPRESSO_SSD_GATE_DECODE` | 1 | Use the native MTLSharedEvent gate if built (`0` disables). |
| `MOESPRESSO_NATIVE_DIR` | unset | Search dir for the native gate extension. |
| `MOESPRESSO_ROUTED_DECODE_SPLIT` | 2 | Output-split factor for the routed decode kernel. |
| `MOESPRESSO_ALLOW_PARALLEL_SSD_STREAMING` | unset | Bypass the single-owner run lock. |
| `MOESPRESSO_SSD_STREAMING_LOCK_PATH` | `/tmp/moespresso-ssd-streaming.lock` | Run-lock path. |
