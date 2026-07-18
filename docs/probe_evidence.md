# Probe: `probe_evidence` from a `source_inventory` + the model weights

`src/moespresso/probe/`

## What the probe is for

The probe is the measurement phase between inventory and optimization. It does not
decide anything; it produces evidence. For every tensor the inventory marked as a
probe target, the probe streams a small weight sample off disk, runs that sample
through a quantize→dequantize round-trip at each candidate bit-width, and scores
how well the round-trip preserves what the weight *does*, using an activation
importance vector supplied by calibration. The result is a per-(bits, group-size)
quality table on each unit: the q-table the optimizer (`optimize.decide` ->
`optimize.allocate`) later turns into a bit allocation.

The split is functional core versus imperative shell:

- **Imperative shell**: `build.py` (orchestration) and `weight_io.py` (the I/O
  edge). These touch the filesystem, choose samples, and assemble the artifact.
- **Functional core**: `quality.py` (pure proxy math) and `roundtrip.py` (pure,
  given the quant libraries). No model, no inventory parsing, deterministic.
- `calibration.py` + `gguf_parse.py` are the calibration provider and the binary
  reader it stands on.
- `deepseek_v4/` (`probe.py`, `codec.py`, `experts.py`) is the model-specific
  probe adapter for DeepSeek-V4 Flash: source-FP4 expert decoding, dense-codec
  quality tables, and the evidence builder the shared entry point dispatches
  to.

The probe never re-parses tensor names or re-resolves GGUF keys. That already
happened in the inventory phase. The probe consumes the inventory's typed fields
(`role`, `kind`, `layer_index`, `projection`, `gguf_keys`, `shape`, `status`) and
carries them forward onto each evidence unit, so the optimizer reads typed fields
rather than splitting names again.

## Building the artifact: `build.build_probe_evidence`

```
build_probe_evidence(inventory, model_dir, calibration=None, *,
                     expert_sample=2, sample_rows=256, seed=42, verbose=False)
```

Flow (the imperative shell):

0. **Dispatch by family.** An inventory whose `family` is `deepseek_v4_flash`
   short-circuits to the DS4 evidence builder in `probe/deepseek_v4/`, which
   decodes the source-FP4 expert groups and emits DS4 units under the same
   artifact contract. The shared flow below is the Qwen-style path.
1. **Resolve calibration** (see next section). Either a `(vectors, identity)` pair
   from a provider, or the explicit uniform escape hatch.
2. `weight_io.scan_offsets(model_dir)`: header-only catalog mapping every tensor
   name to its `TensorHeader` (with byte offsets). No weight bytes read yet.
3. For each inventory tensor with `status == "required"`:
   - look up its header in the catalog (skip if missing);
   - dispatch on `entry["kind"]`:
     - `"expert"` → `_probe_expert` → one or two unit reports (gate/up split if
       the expert tensor is fused);
     - `"affine"` → `_probe_affine` → one unit report.
4. Tally imatrix coverage per kind; emit a **warning** (not an error) for
   incomplete coverage: `probe.no_imatrix` when 0 of N targets of a kind mapped
   to a calibration vector, `probe.partial_imatrix` when some mapped and the
   rest fell back to uniform.
5. `make_artifact("probe_evidence", ...)`: content-addressed, NaN/Inf rejected,
   carrying `units`, `calibration`, `config`, `coverage`, and
   `source_inventory_id` for provenance.

The artifact is `status="valid"` even with a uniform fallback. An uncalibrated
probe is still a valid artifact; whether it is *acceptable* is a policy decision
made downstream (see "fail-closed" below).

### Per-unit fields

Each unit records the inventory's typed fields plus the measurement:

- `source_name`, `kind` (`"affine"`/`"expert"`), `role`, `layer_index`;
- `shape`: **the TRUE tensor geometry** `[out_features, in_features]`, never the
  sampled-row count. This is load-bearing: the optimizer prices bytes from
  `shape`, so recording sampled rows there would under-count expert bytes ~2x and
  collapse any size budget. How much was sub-sampled lives in the separate
  `sampled` field.
- `importance`: one scalar for the optimizer's fidelity weighting
  (`_scalar_importance`): mean of the imatrix vector when calibrated, else the
  RMS of the weight sample;
- `imatrix_mapped`: bool, whether a calibration vector actually keyed in;
- `quality`: the q-table, `{key -> q}`. Affine keys are `"{bits}_{group_size}"`;
  expert keys are `"{bits}"`.

DeepSeek-V4 dense units add codec-decision fields: `dense_codec_quality`
(MX-float q-table keyed `"{mode}_{bits}_32"`, e.g. `mxfp4_4_32`), the unit's
`source_codec`, and `lossless_codecs` (the target codecs that carry the source
representation losslessly). The optimizer's MX-float decision parameters read
these.

## Calibration is a required input: `calibration.py`

The probe weights reconstruction error by how much each input channel actually
drives activations on real data. That per-channel importance vector `h` is the
calibration. The module defines a calibration provider contract:

```
provider(path) -> (vectors, identity)
  vectors:  {gguf_key -> per-input-channel importance h}   # float32, len = in_features
  identity: {kind, name, source, size_bytes, sha256, key_count, sampling}
```

- `vectors` feeds the activation-weighted quality math.
- `identity` is the spec-required **calibration-dataset identity** recorded
  verbatim in `probe_evidence`, so the evidence is pinned to exactly which
  calibration produced it (sha256 + size). Its absence was the audit's #1 gap.

### The imatrix provider: GGUF and legacy `.dat`

`imatrix_calibration(path)` is the concrete provider. Its vector reader,
`read_imatrix_vectors(path)`, dispatches on the file's magic bytes to one of
two supported llama.cpp imatrix container formats:

- **GGUF imatrix** files (the current llama.cpp output format).
- **Legacy `.dat`** imatrix files (the older binary entry list). Entries are
  read one at a time and normalized by their call count. DeepSeek-V4 legacy
  files store each routed-expert tensor as one flat `[expert, input]` entry;
  the reader collapses experts to the per-input mean so the probe gets one
  importance vector per logical expert unit.

Anything else is rejected with a clear unsupported-format error.

The importance per input channel is the diagonal-Hessian activation energy

```
h_j = E[x_j^2]  =  (Σ_e in_sum2[e, j]) / (Σ_e counts[e])
```

For the GGUF form, two cases:

- **Dense** tensors store 1D `in_sum2 [in]` and a single `counts [1]`.
- **Stacked expert** tensors store 2D `in_sum2 [n_experts, in]` with a count *per
  expert*. The corpus-aggregate importance sums `in_sum2` over experts and divides
  by the sum of all expert counts.

A tensor with total count ≤ 0 maps to an all-zero vector (which the quality math
treats as "no per-channel signal" and falls back to uniform, see below). Keys are
GGUF base names like `blk.3.ffn_down.weight`.

The GGUF file is **memory-mapped**; only the requested vector bytes fault in
(`np.frombuffer(mm[...]).copy()`), so a large imatrix never lands in RAM whole.
The copy makes each array outlive the map so `mm.close()` is unblocked.

`calibration.py` also exposes `imatrix_expert_counts(path)`: per-layer routed-
expert usage counters (`blk.N.ffn_gate_exps.weight.counts`), real routing evidence
over the calibration corpus, used as the cold-start hotlist source for residency.
gate/up/down share one router, so gate counts are read. The caller must
verify block indices align with the package's routed layers (fail-closed).

### Uniform fallback: an explicit opt-in that cannot pass as calibrated

`calibration=None` is the only way to get an uncalibrated probe, and it is an
**explicit** escape hatch (synthetic tests, no-imatrix research). When it is used:

- the evidence is stamped `calibration = {"kind": "uniform"}`;
- it declares **no** required feature.

When a real provider is given:

- `calibration` is the provider's identity dict;
- the evidence declares `required_features = ["calibration"]`.

That asymmetry is what keeps an uncalibrated probe from masquerading as a
calibrated one. `"calibration"` is a known feature in the artifact contract
(`core/artifact.py: KNOWN_FEATURES`), and the fail-closed rule says a reader that
does not understand a required feature must refuse the artifact. A calibrated
probe announces the feature; the uniform escape hatch announces nothing. There is
no field a uniform probe could set that would make a "calibration required"
consumer accept it.

Note the layering: the probe core stays format-agnostic. The policy "mjtq
requires calibration" is enforced by the caller that chooses mjtq
(`package.convert`). The probe records the distinction and leaves the policy to
its caller.

There is also a soft signal: per kind, if some targets exist but none mapped to a
calibration vector, `build` appends a `probe.no_imatrix` **warning** to the
artifact's validation list (uniform fallback in effect). This warning leaves
the artifact valid; acceptability is the consumer's call.

## Activation-weighted reconstruction-quality proxy: `quality.py` (pure)

The proxy math lives here. No mlx, no model: given an original weight matrix `W`,
its quantized-then-dequantized reconstruction `Ŵ`, and a per-input-channel
importance vector `h`, score how well `Ŵ` preserves what `W` does:

```
            Σ_j  h_j · ‖ W[:,j] − Ŵ[:,j] ‖²
q  =  1  −  ───────────────────────────────
                 Σ_j  h_j · ‖ W[:,j] ‖²
```

Read column-wise (`j` indexes input channels / columns):

- `col_err[j]  = Σ_i (W[i,j] − Ŵ[i,j])²`  : per-channel reconstruction error;
- `col_energy[j] = Σ_i W[i,j]²`           : per-channel weight energy;
- `q = 1 − (h · col_err) / (h · col_energy)`.

Properties that matter:

- **Uniform `h` reduces it to normalized MSE.** When `h` is all-ones (or, in
  `activation_weighted_quality`, when `h · col_energy ≤ 0`, no per-channel
  signal), the formula collapses to `q = 1 − Σ col_err / Σ col_energy`. So the
  uniform fallback evaluates the same metric with flat weights.
- **A zero weight reconstructs perfectly** (`q = 1.0`): guarded so an all-zero
  energy denominator does not divide by zero.
- **Robust to row sub-sampling.** The sampling factor appears in both numerator
  and denominator and cancels in the ratio, so probing a random subset of rows
  gives the same `q` as the full tensor in expectation. This is why the probe can
  afford to sample.
- **`Q_FLOOR = −1.0` sentinel.** A NaN/Inf or blown-up reconstruction floors at
  `Q_FLOOR` rather than poisoning the artifact (the artifact contract rejects
  non-finite floats outright). `q` is also clamped to be ≥ `Q_FLOOR`.
- An `importance` length mismatch vs `in_features` raises: a programming error,
  not a silent wrong answer. (`build._importance_vector` guards against this
  upstream by falling back to uniform when lengths disagree.)

`cosine(W, Ŵ)` is also provided: a legacy whole-matrix cosine similarity. The
round-trips report both `(cosine, q)`, but the optimizer believes `q`.

## Quantize → dequantize round-trips: `roundtrip.py`

The three format families the probe measures, each a pure function of its
sample (given the quant library is present):

- **affine** (`affine_roundtrip`): mlx affine `quantize`/`dequantize` for
  **non-expert** weights, parametrized by `bits` and `group_size`. Returns a
  float32 reconstruction of the same shape.
- **MLX MX-float** (`mx_float_roundtrip`): mlx `quantize`/`dequantize` in
  `mxfp4` or `mxfp8` mode (fixed group 32, UE8M0 scales) for dense weights.
  The DS4 probe consumes this through its `dense_codec_quality` tables.
- **TurboQuant / TQ** (`tq_reconstruct`): for **stacked MoE experts**, via
  `jang_tools.turboquant`. The full round-trip is: `tq_quantize_weight` → unpack
  packed bits → codebook lookup → scale each row by its stored norm → inverse
  Hadamard rotation (`hadamard_inverse` with the seeded random signs). Returns
  float32, same shape as the (already sub-sampled) input.

Each has a `*_quality(...)` wrapper that runs the round-trip and returns
`(cosine, activation_weighted_quality)`.

Two engineering notes:

- The heavy deps (`mlx`, `jang_tools`) are imported **lazily** with a clear error
  if absent, so the rest of the package imports fine in an environment without
  them.
- Each round-trip explicitly `del`s its intermediates and calls
  `mx.eval(); mx.clear_cache()` (`_flush`) to keep mlx memory bounded, important
  because the probe runs many round-trips back to back.

Which bit-widths/groups are swept lives in `build.py`:

- experts: `EXPERT_BITS = (1, 2, 4)` (TQ);
- affine: `AFFINE_BITS = (2, 3, 4, 5, 6, 8)` × `AFFINE_GROUP_SIZES = (32, 64, 128)`,
  skipping any group size that does not divide `in_features`.

All measurement is fp16-on-disk → float32 in memory (`config.fp16_measurement`).

## Streaming weight bytes: `weight_io.py` (memory-bounded)

The probe must run in a few GB on a 35B model, so it **never materializes a full
tensor**. `weight_io` is the I/O edge: it reads safetensors headers, seeks to a
tensor's byte range, and converts only the bytes it needs to float32. No mlx here.

- `scan_offsets(model_dir)`: header-only catalog
  (`name -> TensorHeader{shape, dtype, shard, header_size, begin, end}`) across all
  `*.safetensors` shards, via `inventory.safetensors_header`.
- `_read_range(model_dir, h, byte_start, nbytes)`: the one primitive, open the
  right shard, `seek(header_size + begin + byte_start)`, read exactly `nbytes`.
- **BF16 is carried as `uint16` until the last moment** to avoid a 2x bloat, then
  shifted into float32 (`u16 << 16` reinterpreted) by `_bytes_to_float32`. F16/F32
  use native numpy views. `_bytes_to_raw` preserves the storage dtype where the
  caller wants raw bytes (used by the streaming-convert paths).

Sampling readers used by the probe:

- `load_2d_sample(model_dir, h, sample_rows, seed)`: for **2D non-expert**
  weights. Rows are contiguous on disk and these tensors are small, so it reads the
  whole tensor then indexes up to `sample_rows` random rows. Row-subsampling
  preserves per-column structure, which is exactly what the per-column importance
  vector needs.
- `load_expert_sample(model_dir, h, n_experts, seed)`: for **stacked-3D MoE**
  tensors `[E, rows, cols]`. Seeks directly to each chosen expert's byte offset, so
  peak memory covers only the sampled experts. Returns the chosen
  experts concatenated along rows: `[n_sampled*rows, cols]`. **Fine-grained MoE:**
  experts within a layer are near-equivalent, so the probe samples at most
  `expert_sample` (default 2) and records the count: one sample resident at a time.
- `split_fused_gate_up(sample, n_sampled)`: a fused gate_up tensor stacks gate
  over up within each expert's rows; this splits the concatenated sample back into
  (gate, up) halves per expert, so each projection is probed and priced
  separately.

`_probe_expert` then sub-samples rows further (down to `sample_rows`) to keep each
round-trip cheap, while keeping the recorded `shape` at the *true* per-projection
geometry (`out_features = rows // 2` for a fused half).

Streaming helpers used elsewhere (convert/package) and kept memory-bounded the
same way: `iter_row_chunks` (bands of a 2D tensor under a byte cap, for
vocab-sized embed/lm_head), `iter_experts` (one expert at a time), the
selected-row/expert readers (`load_2d_rows[_raw]`, `load_3d_rows[_raw]`,
`sample_indices`), and `load_full[_raw]` for tiny structural tensors (norms, SSM
state) that need no streaming.

## The progressive GGUF parser: `gguf_parse.py`

`GGUFBufferParser` is a stdlib-only, progressive parser of the GGUF binary
container, fed a growing `bytearray` chunk by chunk. It exists so calibration can
read an imatrix **header** without faulting the (much larger) tensor-data section
into memory: `calibration._parse_header` mmaps the file, feeds 1 MiB chunks to the
parser, and stops as soon as `is_complete()`: header + all KV pairs + all tensor
infos parsed.

Mechanics:

- `feed(data)` appends bytes; `try_parse()` consumes as many *complete* records as
  the buffer currently allows, in order: header → KV pairs → tensor infos. An
  incomplete record leaves `_pos` untouched (each `_try_*` resets `_pos` on a short
  read and returns `None`), so partial chunks never corrupt state. The next
  `feed` + `try_parse` resumes cleanly. This is what "progressive" buys.
- It parses the 24-byte header (magic `GGUF`, version ≥ 2, tensor count, KV count),
  then KV pairs over the full GGUF value-type table (scalars, strings, and nested
  arrays via recursive `_try_read_value`), then tensor infos
  (`name, n_dimensions, dimensions, type_id, offset`).
- **Safety limits, fail-closed:** invalid magic / unsupported version raise;
  `MAX_KV_COUNT = 100_000` and `MAX_ARRAY_LEN = 1_000_000` reject pathological
  counts before allocating.
- `total_consumed()` returns the parser position; calibration uses it to compute
  the tensor-data offset. **GGUF aligns the tensor-data section to 32 bytes** past
  the header: `data_offset = ((total_consumed() + 31) // 32) * 32`. Each tensor's
  absolute byte start is then `data_offset + tensor_info.offset`.

GGUF stores dimensions innermost-first, so an on-disk `in_sum2` of shape
`[in, n_experts]` is reshaped to the logical `[n_experts, in]` row-major in
`calibration._iter_channel_sums`. The `_pairs` helper matches each `<base>.in_sum2`
with its following `<base>.counts` to emit the (offsets, shapes) the importance
math consumes.

## How it fits the pipeline

```
source_inventory ─┐
                  ├─► build_probe_evidence ─► probe_evidence ─► optimize.decide ─► optimizer_decision
model weights ────┤        (this module)        (q-tables)
calibration ──────┘
(imatrix provider)
```

- **Consumes:** the `source_inventory` artifact (typed targets), the safetensors
  shards (streamed), and a calibration provider's `(vectors, identity)`.
- **Produces:** the `probe_evidence` artifact, with per-unit `quality` tables +
  `importance`, the pinned `calibration` identity, `config`/`coverage`, and the
  `source_inventory_id` link.
- **Feeds:** the optimizer, which reads only typed fields (`role`,
  `layer_index`, `projection`, true `shape`, `importance`, `quality`) and never
  re-parses names.
