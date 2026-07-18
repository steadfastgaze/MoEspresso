# The correctness ladder

The correctness subsystem (`src/moespresso/correctness/`) is the
implementation-time validation layer: cheap, model-free checks that prove the
converter/writer path produced a package mathematically consistent with what the
optimizer intended and what the runtime will reconstruct, before anyone spends
hours loading or serving a full 35B model.

## The invariant

**Correctness requires token or logit identity.** Greedy decoded text can look fine while
the underlying logits are corrupted; a transform, sidecar, codec,
or sanitizer can be wrong in a way that survives a casual read of the output. The ladder
exists because "the text looked okay" is the weakest possible evidence. Every rung asks
whether the stored bytes reconstruct to the declared math (an exact relation derived from
source tensors plus the family's declared transforms), never whether a sample reads well.

## The reference trust boundary

A check is only useful if its reference is not the same code path as the thing under test.
Every piece of evidence records its reference strength:

- `independent`: MoEspresso-owned reference math that does not call the production writer,
  decoder, sanitizer, or runtime fast path it verifies (e.g. the project's TQ stored-array
  reference, the fused gate/up split, the conv1d/norm shape predicate).
- `external_codec`: a stable external library that also owns the format (MLX
  `dequantize` for affine sidecars). This validates MoEspresso's source selection, manifest
  metadata, sidecar presence, and package locations; it does *not* independently validate
  the external library's quantization math.
- `shared_code`: imports the same codec/sanitizer/loader as production. Useful as a smoke
  check, but it can never be the sole evidence that clears the code it shares.

A rung cannot clear the same production code it imports on both sides of a comparison.

## The ladder: named rungs

The implemented rungs are coded `L0`, `L0b`, `L1`, `L2` (the short codes are kept; the
full DS4-derived ladder runs to L8, but L3-L8 currently exist only as designs). Each rung emits a
`correctness_evidence` artifact whose `status` is `invalid` if any finding is blocking and
`valid` otherwise. A bad package's evidence is unmistakably invalid, never "valid with
notes." `make_correctness_evidence` in `ladder.py` does this wrapping.

### L0: static contract check (`l0_static_contract`, `ladder.py`)

Pure: inputs are the family `architecture_profile`, the source `inventory`, and the
package `manifest`. No tensor IO. It catches wrong *ownership* before any weight is loaded.
Plainly, it proves that every source tensor has a declared owner and that the manifest's
chosen on-disk format matches that owner. Specifically:

- **Every non-excluded source tensor is owned.** An owner is one of: a quant owner declared
  in the profile's `role_quant` (`affine` / `tq` / `fp16`), `passthrough` for a declared
  unquantized structural tensor (norms, SSM state, conv1d, carried fp16), or `excluded`
  (a namespace the profile or the inventory marked out-of-scope, e.g. `vision`, `mtp`). A
  tensor with no declared owner is a blocking `unowned_tensor`. Nothing may be implicitly
  handled.
- **Every owned tensor is carried** in the package manifest (else it would be silently
  dropped → `tensor_not_carried`).
- **No out-of-scope tensor ships**: an excluded-namespace or `kind='excluded'` tensor
  present in the manifest is `excluded_tensor_carried`; a manifest tensor the inventory
  never declared is `undeclared_package_tensor`.
- **Each stored format matches its declared owner.** This is the headline check: a tensor
  whose declared owner is (say) `affine` but stored as `fp16` is `quant_kind_mismatch`. It
  catches the `in_proj`-as-fp16 class of bug. Checks run per *manifest entry*. They are never
  collapsed by source name, so a fused `gate_up_proj` (which appears twice, as gate and up) cannot let
  a bad half hide behind a good one.

A malformed entry (no `source_name`) becomes a finding, never a crash.

### L0b: header storage-contract check (`l0b_norm_shift_contract`, `ladder.py`)

Header-only: reads safetensors headers from the package shards, no weight bytes and no
model. It enforces the conv1d / RMSNorm-shift **storage-versus-runtime** contract,
guarding the failure mode where storing conv1d pre-transposed suppresses the coupled
norm shift, so norms load ~1.0 too low and the model emits garbage.

The contract: norms are *stored unshifted*, and the runtime sanitizer adds a `+delta`
(`+1.0`) shift *if and only if* the conv1d "trigger" holds. The trigger is a shape
predicate: for qwen3_5_moe, the source layout `[out, 1, k]` (last dim `!= 1`) fires the
sanitizer; the broken layout `[out, k, 1]` (last dim `== 1`) does not. So L0b must **not**
demand shifted norms on disk (that would falsely reject a correct package). Instead it
checks that the *stored conv1d shape satisfies the trigger* so the required runtime shift
will actually fire. It blocks (`norm_shift_suppressed`) only when a required shift is
suppressed because the on-disk shape kills the trigger: i.e. the norms would load ~1.0 too
low and the model would emit garbage.

`expect_conv1d` says whether this package should carry a conv1d at all. The coupling is
`required` for the family, but a full-attention-only or reduced/smoke build legitimately
has none. So when `expect_conv1d=False`, a legitimately *absent* conv1d does not block,
while a *present* one is still shape-checked (the layout bug cannot hide in a subset). When
`expect_conv1d=True` (default, strict) and the profile requires the coupling but no conv1d
tensor is found, that blocks (`required_conv1d_absent`). The required shift is unprovable.

### L1: tensor-reconstruction check (`l1_tensor_reconstruction`, `reconstruct.py`)

Samples rows/experts directly from source and package shards: no model graph, no full
package serve. Heavy references (MLX) are imported lazily so L0/L0b stay lightweight. It
proves that the stored package data reconstructs to what the writer intended, verifying the
*storage-versus-runtime relations* hold, including the norm-shift. Sampling is deterministic
(seeded) and recorded in the evidence; high-risk roles (`lm_head`, `embed_tokens`,
`in_proj_a/b`, `conv1d`, `norm`, `gate_up`) are prioritized. A "passed because nothing was
sampled" result is forbidden: if the manifest claims a format but L1 sampled none
successfully, that is blocking (`no_samples`).

By tensor class:

- **Structural passthrough (fp16)**: package shape must equal source shape, and stored
  values must equal the source after fp16 conversion within `passthrough_max_abs` (2e-3).
  For norms this encodes the storage relation `stored == source` (the `+1.0` runtime shift
  is *derived for reasoning, never required at storage*); for conv1d it confirms stored
  values match the source in source layout (L0b already guards the layout).
- **Affine**: package carries `.weight` / `.scales` / `.biases`; manifest `bits` /
  `group_size` are valid; sampled rows dequantized through MLX `dequantize` (`external_codec`)
  reconstruct to the source rows within `affine_relative_rms` (1.25 relative RMS).
- **TQ experts**: package carries the layer bundle with per-projection geometry; the bundle's
  stored `bits` must agree with the manifest (`tq_bits_mismatch`); sampled expert rows decode
  through the project-owned TQ reference (`independent`) and reconstruct within
  `tq_relative_rms`. Fused `gate_up` source halves must map to the declared `gate`/`up`
  package projections. The bundle row read is the exact byte range the runtime preads, so L1
  verifies the loader's actual contract.

Blocking on any missing/extra/shape-mismatched tensor or sidecar, nonfinite reconstruction,
mismatched `tq_bits`, missing gate/up half, or reconstruction error over the per-format
tolerance. Tolerances are per-format and recorded in evidence, never one global magic
number.

### L2: micro-golden evidence (`l2_micro_goldens`, `goldens.py`)

Tiny, deterministic, no model, no package serve. These pin the project-owned primitives
that L1 depends on, each with `independent` reference provenance:

- **TQ unpack**: 1/2/4-bit uint32-packed index unpacking returns the expected indices.
- **Hadamard inverse**: `hadamard_inverse(hadamard_rotate(x)) == x` within 1e-6.
- **Fused gate/up split**: fixed tiny source arrays with visibly distinct gate/up halves
  split into the correct fixed halves (a self-check of the helper against itself is
  explicitly avoided).
- **conv1d / norm-shift trigger**: the shape predicate distinguishes good `(8,1,4)` from
  bad `(8,4,1)`, duplicating the historical failure in tiny form.
- **affine sidecar shapes**: the declared MLX affine sidecar shape rule
  (weight/scales/biases) holds for a tiny weight.

Any failure is blocking (`golden_failed`). Policy: fast paths are never accepted because
generated text looks plausible; a primitive change must pass its micro-golden first.

## The project-owned TurboQuant reference (`tq_reference.py`)

A small, deliberately simple decoder for stored MJTQ TQ arrays, **independent of**
`jang_tools`. It decodes directly from the frozen format facts so TQ reconstruction
evidence can be `independent` rather than `shared_code`. It is a correctness reference for
sampled rows and does not serve inference. It owns:

- `unpack_tq_indices`: uint32-packed codebook indices, low bits first, `32 // bits` values
  per word.
- `compute_codebook`: the Lloyd-Max scalar codebook for TQ's rotated unit-vector
  coordinates at `(in_features, bits)` (cached).
- `generate_random_signs`: deterministic ±1 signs (seeded) for the randomized Hadamard
  rotation.
- `hadamard_rotate` / `hadamard_inverse`: the blockwise (power-of-two decomposition)
  randomized Hadamard transform and its inverse.
- `tq_decode_rows`: the full path: unpack indices → codebook lookup → scale by fp16 row
  norm → inverse randomized Hadamard → float32 `[rows, in_features]`.

L1 uses `tq_decode_rows`; L2 pins `unpack_tq_indices` and the Hadamard round-trip.

## The convert-time correctness gate (`gate.py`)

The rungs above were built as standalone evidence first; the gate is the wiring that turns
them into a build refusal. `run_convert_gate` runs L0, L0b, L1, and L2 against a
**freshly written** package, collects each rung's `correctness_evidence`, and returns a
`GateResult` whose `passed` is `False` iff any rung produced a blocking finding.

It is **format-agnostic**: the caller (`package/convert.py`) resolves the family's
`architecture_profile` from the model config. If no profile is registered, the gate is
reported `skipped`. The caller warns loudly and proceeds ("package written unverified");
an unprofiled family is never silently passed and never wrongly blocked. The caller also
derives `expect_conv1d` from the model's layer types (True only when linear-attention layers
exist) so L0b does not block a full-attention-only or smoke build that legitimately has no
conv1d.

On a blocking finding the build is **refused**: the convert raises and the package bytes
stay on disk for inspection, with every rung's evidence written under the package's
`correctness/` directory (so both a blocked and an allowed-through convert are always
inspectable). This is overridable by an explicit flag (`--allow-incomplete`) which writes
the package anyway and records the blocking findings, mirroring the allocation health gate's
`--allow-unhealthy`. The override is for research only.

This is the check that catches the stored-layout gibberish class at build time, without
loading or serving the full model.

## What this is not

The ladder does not replace the optimizer's allocation-risk work (whether a bit allocation
is good enough) or the health gate (whether an allocation has collapse signatures). It asks
a separate question: does the package/runtime path *implement the declared math*. A package
can have a healthy allocation decision and still be wrong because a transform, sidecar,
codec, or sanitizer is wrong. That is what correctness evidence exists to catch.

L1/L2 also do not answer whether the model is coherent, whether external inference kernels
are correct, or whether logits match a known-good runtime. Those are the deferred upper
rungs (L3 layer-local diffs, L4 frontier checks, L5 logit sketches, L6 fixed continuations,
L7 capability canaries, L8 full serve). Full serve remains the final integration proof. It
confirms a corrected path; it is never the first place a mismatch is discovered.

Above the generic ladder, each shipped family carries its own served quality gates,
because quality ladders are model-specific:

- **DeepSeek-V4** (`correctness/deepseek_v4/quality.py`, `moespresso-ds4-quality`):
  Q0 renderer/tokenizer goldens (no model forward), Q1 exact-answer gate, Q2
  teacher-forced NLL, Q3 long-context gate, plus the replay/diff debug tools beside
  them.
- **Ornith** (`correctness/ornith/`, `moespresso-ornith-gate`): gate v2, combining
  hard-reasoning items, sandboxed self-verifying agentic coding tasks, and exact-scored
  long-context questions.
- **Qwen 35B** (`correctness/qwen35/`, `moespresso-qwen35-hard-questions`): the
  exact-answer package comparison harness.

`correctness/environment.py` records the installed mlx wheel tag in gate evidence: the
wheel variants of one mlx version are distinct numeric lattices that move knife-edge
anchors, so an anchor shift stays attributable to the environment instead of surfacing
as a quality mystery.
