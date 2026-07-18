# Inventory subsystem

`src/moespresso/inventory/` is the first phase. It reads a model checkpoint on
disk (or a remote Hugging Face repo) and writes a `source_inventory` artifact
that every later phase consumes. Only the safetensors headers get read; the
weight bytes are left on disk. Every tensor name is resolved exactly once by the
inventory phase before downstream code sees it.

That single resolution is what the rest of the system relies on. Once a tensor is
classified, the probe, optimizer, converter, packager, and runtime read the typed
entry, and the raw name is never split again. If a name-mapping question has an
answer, it lives under `inventory/`: shared/Qwen-style rules live in `roles.py`;
model-specific naming contracts live under model subpackages such as
`deepseek_v4/roles.py`.

## Files

| File | Role |
| --- | --- |
| `safetensors_header.py` | The only place that opens a safetensors file. Reads the JSON header, never the weight bytes. |
| `roles.py` | Shared/Qwen-style source→role and role→imatrix-key resolver helpers. |
| `deepseek_v4/roles.py` | DeepSeek-V4 source→role, GGUF-key, and module-path resolver. |
| `deepseek_v4/static.py` | DeepSeek-V4 static checkpoint validator. |
| `build.py` | One pass over headers → a typed `source_inventory` artifact. The "resolve once" gate. |
| `architecture_profile.py` | The model-family correctness contract: per-role quant ownership, text-only scope, transforms, exclusions. |
| `hf_inspect.py` | A remote Hugging Face header inspector CLI (stdlib HTTP Range requests; no SDK, no download). |

## Real inputs

The subsystem consumes two kinds of real input and nothing else:

1. **Safetensors shards.** The model's weight files (`model-00001-of-*.safetensors`),
   indexed by `model.safetensors.index.json` when present, or globbed otherwise.
   Only the leading JSON header of each shard is read. The header maps
   `tensor name -> {dtype, shape, data_offsets}`. From it the inventory learns
   every tensor's name, shape, dtype, and which shard it lives in. The weight
   payload after the header is never opened by this subsystem.

2. **GGUF imatrix keys.** An optional set of llama.cpp importance-matrix tensor
   keys (e.g. `blk.12.ffn_gate_exps.weight`, `blk.3.attn_q.weight`). These come
   from a calibration `.gguf` produced for the same model family. When supplied,
   the inventory checks that every imatrix key it *resolved* for a quantizable
   tensor is actually present in the real key set: the guard against a silent
   name-mapping regression.

## Reading headers only: `safetensors_header.py`

A safetensors file is laid out as an 8-byte little-endian header length, then a
JSON object, then the raw weight region. To learn the shape and dtype of every
tensor, you only need the JSON header. This module reads exactly that and stops:

- `read_header(path)`: parse one shard's JSON header (drops `__metadata__`).
- `scan_headers(model_dir)`: every tensor across all shards as `TensorHeader`
  records. Uses the index's `weight_map` to find shards if present, else globs
  `*.safetensors`. This is the function `build_inventory` calls.
- `read_headers_with_offsets(path)`: same, but carries each tensor's byte
  offsets (`header_size`, `begin`, `end`) so a *later* phase (the probe) can
  stream the actual weight bytes. The inventory itself does not need offsets, so
  `scan_headers` leaves them at zero.
- `read_shard_metadata(path)`: the shard's `__metadata__` string map (used by
  the packaged-bundle format to carry per-layer expert geometry header-only).

`TensorHeader` is a frozen dataclass carrying `name`, `shape`, `dtype`, `shard`,
and the optional offset fields. Dependency-free: no mlx, no torch. Skipping the
weight bytes keeps the inventory cheap on a model far larger than RAM, and lets
the remote inspector run over HTTP Range requests.

## Resolve once: shared and model-specific role resolvers

The inventory phase owns tensor-name mapping. A raw tensor name maps to two
things:

1. an internal **role**: a stable vocabulary the rest of MoEspresso uses
   (`attn.q_proj`, `moe.expert.gate_up`, `ffn.down_proj`, `norm.final`, …);
2. for quantizable weights, a **GGUF imatrix key**: the llama.cpp tensor name
   used to look up calibration importance.

This is where the "resolve once" rule is enforced. After `build.py` calls the
resolver and writes a typed entry, no downstream phase splits a tensor name on
`.` again. A new model family adds or uses an inventory model subpackage when
its naming contract differs materially from the shared Qwen-style layout;
nothing downstream changes.

`roles.py` contains the shared/Qwen-style resolver: Qwen MoE full-attention and
linear-attention layers, stacked experts, shared experts, dense FFN, and common
stacked-expert package-path helpers.

`deepseek_v4/roles.py` contains the DS4 resolver: DS4 FP4 source experts,
codec-scale companions, dense/shared/control roles, GGUF dense keys, and the
JANG DS4 module-path mapping used by sidecar/package generation.

### Role vocabulary

The shared resolver carries small, explicit mapping tables keyed by the tensor's suffix
after `...layers.N.`:

- **Attention** (`ROLE_ATTN`): `self_attn.{q,k,v,o}_proj.weight` → `attn.*`.
- **Linear-attention / SSM** (`ROLE_SSM`): `linear_attn.in_proj_{qkv,z,a,b}` and
  `out_proj` → `ssm.*`.
- **MoE shared/router** (`ROLE_FFN_SHARED`): `mlp.gate` → `moe.router_gate`,
  shared-expert gate/up/down and `shared_expert_gate`.
- **Dense FFN** (`ROLE_FFN_DENSE`): `mlp.{gate,up,down}_proj.weight` → `ffn.*`.
  Note `mlp.gate.weight` (the MoE router) and `mlp.gate_proj.weight` (the dense
  gate projection) deliberately do not collide.
- **Global** (`ROLE_GLOBAL`): `model.language_model.embed_tokens.weight` and
  `lm_head.weight`.

Stacked MoE experts are 3D tensors matched by a dedicated regex
(`_STACKED_EXPERT_RE`) against `...layers.N.mlp.experts.<proj>`. `parse_stacked_expert`
returns `(layer, short_proj)` where `short_proj` is `gate_up` or `down`.
`expert_role` builds `moe.expert.<proj>`.

### Imatrix-key resolution

The resolver also owns the **role→imatrix-key** direction, with key families
verified against the real 510-key set of the production Qwen3.6 model:

- `non_expert_gguf_key(name, layer_types)` → `blk.<layer>.<fam>.weight`, where
  `<fam>` is chosen from the dense-FFN, shared-FFN, full-attention, or
  linear-attention key tables. **Layer kind matters here:** for attention
  suffixes the key family depends on whether layer `N` is `full_attention`
  (→ `attn_q`/`attn_k`/…) or `linear_attention` (→ `attn_qkv`/`ssm_alpha`/…),
  so `layer_types` is threaded in. When layer types are unknown the suffix sets
  are disjoint, so it safely tries both.
- `expert_gguf_keys(layer, projection)`: a fused `gate_up` expert maps to **two**
  keys (`ffn_gate_exps` + `ffn_up_exps`); `down` maps to one.
- Returning `None` for a key means *"no imatrix entry is expected"*
  (`embed_tokens`, `lm_head`), which is deliberately distinct from a mapping bug.
  Because the inventory counts coverage, a silently-zero mapping cannot pass.

### Structural roles

`structural_role(name)` decides whether an un-quantized tensor must still travel
into the package as **passthrough**: per-layer norms, q/k norms, the SSM norm,
and the SSM state params (`A_log`, `dt_bias`, `conv1d`), plus the final
`model.norm`. It returns `None` for anything outside the text graph (`mtp.*`,
`visual`/`vision`, rope inverse-frequency, biases) so those stay dropped. This
is the single place that decides "this un-quantized tensor must be carried."

### Sanitized output paths

Two helpers compute the on-disk module paths the packaged/runtime layout uses,
both derived from the same sanitized head so the conventions can never drift:

- `switch_mlp_key(source_name, projection)`: rewrites a fused expert source
  (`...mlp.experts.gate_up_proj`) to the TQ-kernel's expected
  `...switch_mlp.<proj>_proj`, also applying mlx_lm's
  `model.language_model.` → `language_model.model.` rename.
- `switch_mlp_bundle_prefix(source_name)`: the per-layer bundle prefix
  (`...switch_mlp.experts`) for the packaged bundle format.

## Building the artifact: `build.py`

`build_inventory(model_dir, layer_types=None, imatrix_keys=None, family=None)`
scans the shard headers and calls `build_inventory_from_headers`, which makes
one pass and classifies each tensor via `_classify`. The `family` argument
gates model-specific classification and is recorded on the artifact.

When `family == "deepseek_v4_flash"`, the classifier dispatches to
`deepseek_v4.roles.tensor_role()` and records DS4-specific kinds such as
`expert_source` and `codec_scale`; an unknown DS4 tensor is a **blocking**
`inventory.unknown_tensors` validation (the DS4 naming contract is closed),
where the shared path records unknowns as a warning. For shared/Qwen-style
inventories, the classifier applies, in order:

1. **Structural passthrough first.** `roles.structural_role(name)` is checked
   *before* the skip list, because the skip-substring list below would otherwise
   drop norms/conv/bias tensors that the graph genuinely needs. These become
   `kind: "passthrough"`, `status: "required"`, with empty `gguf_keys`. `mtp`/
   vision names return `None` here and are dropped.
2. **Skip list.** A tensor whose name contains any of `norm`, `bias`, `rotary`,
   `rope`, `conv`, `_scale`, `vision`, `visual`, `audio`, `image`, `mtp` (and was
   not claimed as structural above) is skipped.
3. **Stacked expert.** A 3D tensor matching the expert regex becomes
   `kind: "expert"` with its layer, projection, and one-or-two resolved
   `gguf_keys`.
4. **2D affine weight.** Any remaining 2D weight is resolved to a non-expert role
   and (optionally) one imatrix key, `kind: "affine"`. A 2D weight with no
   recognized role is still recorded but flagged `role: "unknown"`,
   `status: "unknown"`.
5. All remaining shapes are dropped.

Each entry records: `source_name`, `role`, `kind`, `layer_index`, `shape`,
`dtype`, `shard`, `gguf_keys`, `status` (plus `projection` for experts).

The artifact carries `counts` (`expert` / `affine` / `expert_source` /
`codec_scale` / `passthrough` / `unknown` / `total`), the `layer_types` it was
built with, its `family`, and `imatrix_coverage`.

### Imatrix coverage as a regression guard

When `imatrix_keys` (the real GGUF key set) is supplied, the builder walks every
resolved `gguf_keys` entry and tallies `resolved` / `present_in_imatrix` /
`absent`. Any resolved key that is *not* in the real set raises a
`warning`-level `imatrix.key_absent` validation pointing at the offending source
tensor. A nonzero `unknown` count likewise raises `inventory.unknown_tensors`.
The artifact `status` is `valid` unless a blocking validation is present. This is
how a silent name-mapping regression (where the resolver invents a key the
calibration file has never heard of) is caught instead of quietly degrading
quantization quality.

Pure except for reading headers at the edge: no weight bytes, no mlx.

## The family contract: `architecture_profile.py`

A `source_inventory` records what tensors a checkpoint has. An
`architecture_profile` records what a correct package of that model family has to
satisfy, and the correctness ladder checks a package against it. The profile is a
per-family declaration: roles, quant ownership, transforms, layer kinds, and
exclusions. Adding a family means adding a profile, with no schema change. Each
profile is emitted as a standalone `architecture_profile` artifact, currently not
wired into the convert/serve/verify path.

### Per-role quant ownership

The `role_quant` map covers every role in the inventory's vocabulary, assigning
each one to a quant regime:

- **`tq`**: TurboQuant. Owned exclusively by MoE experts (`moe.expert.*`).
- **`affine`**: MLX affine quantization. Every other 2D non-expert weight:
  attention projections, the SSM `in_proj_*`/`out_proj`, dense FFN gate/up/down,
  shared-expert gate/up/down, embeddings, and `lm_head`. Notably the SSM
  `in_proj_a`/`in_proj_b` deliberately use affine quantization. Keeping them
  fp16 produced a measured correctness regression.
- **`fp16` passthrough**: only the discrete routing gates stay full precision:
  `moe.router_gate` and `moe.shared_expert_gate`. These are the gates whose
  argmax routing decision is too sensitive to quantize.

Separately, `structural_passthrough` lists the structural tensors carried
verbatim because the graph needs them. They are not a quant choice: the norms,
q/k norms, SSM norm, `A_log`, `dt_bias`, and `conv1d`.

### Text-only scope and exclusions

Every Qwen3.5 profile declares `modality: "text"` and an `excluded_namespaces`
map naming the source scopes that are *not* served:

- `visual` / `vision`: the real Qwen vision tower is `model.visual.*`; a
  text-only package does not serve it. The exclusion token stays in lockstep with
  the inventory skip/structural logic in `roles.py`.
- `mtp`: multi-token-prediction draft layers are not part of the base text
  forward. (MTP presence is itself a sanitizer norm-shift trigger; excluding it
  makes the conv1d shape predicate the *sole* shift trigger.)

Because the profile spells these exclusions out ahead of time, the inventory
reports `model.visual.*` and `mtp.*` as known exclusions rather than `unknown`
text drift. A genuine unknown 2D text weight still surfaces as a warning.

### Source→runtime transforms

The profile declares the `transforms` that turn stored package tensors into the
runtime graph. The load-bearing pair for the Qwen3.5 hybrid stack:

- **`conv1d_layout`**: the SSM `conv1d.weight` MUST be stored in source shape
  `[out, 1, k]`. The runtime sanitizer transposes it to `[out, k, 1]`. The stored
  shape's last dim being `!= 1` is the structured `sanitizer_trigger` the ladder
  checks mechanically.
- **`rmsnorm_shift`**: RMSNorm weights are *stored unshifted*; the runtime
  weight is `source + 1.0`. Storage and runtime are separate relations: the
  ladder must check `storage == source` **and** the presence of the trigger. The
  stored norm must remain unshifted. This is the contract whose
  violation produced pre-fix gibberish: storing conv1d pre-transposed suppresses
  the coupled `+1.0` shift, leaving the norms too low. The shift is `coupled_to`
  the conv1d trigger and is `required` for this family.

The MoE profile additionally declares `key_rename`
(`model.language_model.*` → `language_model.model.*`) and `expert_fused_gate_up`
(the fused `[E, 2*moe, hid]` expert tensor splitting into `switch_mlp.gate_proj`
+ `up_proj`). The dense profile shares the conv1d/RMSNorm contract but declares
no expert transform at all: no fused gate_up split, no switch_mlp rewrite, no
router-gate or shared-expert-gate preservation. The dense profile is kept
separate for exactly this reason. It carries the same Qwen3.5 hybrid sanitizer
contract while the MLP contract differs.

### Family resolution

The profile module owns mapping a source config to a family and then to a
profile, generically:

- `family_of(config)` resolves a `model_type` (top-level and/or `text_config`) to
  a family id. MoE tokens (`qwen3_moe`, `qwen3_5_moe`, `qwen3_5_moe_text`) win
  first. A wrapped dense Qwen3.5 (`qwen3_5` + inner `qwen3_5_text`) resolves to
  `qwen3_5_dense` **only when no expert fields are declared**. An expert-bearing
  `qwen3_5` config is not treated as dense. A bare `qwen3` config does not
  accidentally resolve to dense Qwen3.5.
- `profile_for(config)` returns the registered profile, or `None` (not a guess)
  when the family is unknown, so a convert gate can *skip with a loud warning*
  rather than apply the wrong contract.

Registered families: `deepseek_v4_flash`, `qwen3_5_moe`, `qwen3_5_dense`, and
`synthetic_dense` (a minimal dense family with no experts and no SSM, present
to prove the schema is generic). `family_of` resolves the `deepseek_v4` model
type to `deepseek_v4_flash`. A new family registers its builder in `PROFILES`
and its `model_type` aliases once the profile has been written from the real
model's facts.

## Remote inspection: `hf_inspect.py`

A CLI for inspecting a remote Hugging Face model *without downloading it and
without a Hugging Face SDK dependency*. It uses only stdlib HTTP and **Range
requests**, applying the same "headers only" discipline as the local path, but
over the network.

```
python -m moespresso.inventory.hf_inspect <hugging-face-url>
```

- For a **safetensors repo**, it fetches `config.json` and the
  `model.safetensors.index.json` over plain HTTP, then for each shard issues two
  Range requests: the first 8 bytes for the header length, then the JSON header
  itself. It prints the config summary (architecture, model_type, sizes, expert
  counts, tie flag), per-shard tensor counts, a dtype breakdown, and a compressed
  tensor listing that collapses numeric layer indices into ranges (so 96 layers'
  worth of `blk.{}.attn_q` collapse to one line). Shard headers are fetched
  concurrently with a small thread pool.
- For a **`.gguf` URL** (e.g. a calibration imatrix or quantized model) it streams
  metadata and tensor info chunk-by-chunk with a buffer parser, capping total
  bytes fetched (50 MB) and bailing if it stops making progress.

`inspect_url` validates the host is Hugging Face, normalizes `/blob/` to
`/resolve/`, and dispatches by file type. Use it to check what tensors and config
a remote repo has before bringing it through the local inventory.

## How it fits together

```
safetensors shards ──► scan_headers ──► build_inventory_from_headers
                                              │  (resolve ONCE via inventory roles)
GGUF imatrix keys ────────────────────────────┤  (coverage / regression guard)
                                              ▼
                                       source_inventory artifact
                                              │
                       probe · optimizer · converter · packager · runtime
                              (read typed entries; never re-parse names)

source config ──► family_of ──► profile_for ──► architecture_profile artifact
                                                   (family correctness contract,
                                                    checked by the correctness ladder)
```

The inventory and the profile are two independent artifacts that come together
downstream. The inventory enumerates the tensors on disk. The profile sets out
what a correct package of that family looks like. With both in hand, every later
phase works from typed, already-resolved facts and checks itself against the
family contract.
