# Published comparison protocol

The README tables compare complete engine, artifact, and cache stacks. This
document defines the controls and evidence required for an independent
repetition. The cross-engine sections that follow, from artifact pinning
through engine seams, cover Ornith. DeepSeek-V4-Flash has a MoEspresso-only
drafter certification at 3,844 prompt tokens and a separate 37,000-token decode
comparison across independently packaged 0731 artifacts in MoEspresso, DS4, and
llama.cpp. The quality protocol uses the same three product stacks. MoEspresso
does not ship a cross-project benchmark framework, copies of third-party
runners, exact prompt-token fixtures, or benchmark-only measurement adapters.

Keep raw results and any temporary adapters in a separate work directory. Large
downloads happen before the measurement session and run one at a time.

## Versions and artifacts

The Ornith cross-engine matrix in this document was measured with these engine
versions:

| Engine | Version |
|---|---|
| MoEspresso | `1.0.0` |
| mlx-kquant | `0.3.0` at `e165cafafa149493d298871c610e29e95ffa8f10` |
| mlx-lm | `0.31.3` with MLX `0.31.2`, source `15b522f593b7ca5fbc0cac6f7572d40859d2d8fe` |
| oMLX | `0.5.1` |
| llama.cpp | `6eddde06a4f25d55d538b5d15628dcc2b6882147` |

The mlx-kquant revision is the public commit whose source tree matches the
benchmark build.

The current release pins `mlx-kquant` 0.3.0 in `pyproject.toml` and `uv.lock`,
from `https://github.com/steadfastgaze/mlx-kquant.git` at
`6fbfd4f5c925c7ce8dc8239ac7d8e2783d49f089`. That extension serves the K-quant
tensors of the Ornith package. The row above is a different commit of the same
0.3.0 line, recorded because it is the tree the measured build used.

The model inputs of the matrix were pinned to immutable Hugging Face revisions:

| Engine | Artifact |
|---|---|
| MoEspresso | `steadfastgaze/Ornith-1.0-35B-Q4_K_M-MoEspresso@c67d34262a258f815789a2018341317c755c45a6` |
| llama.cpp | `bartowski/deepreinforce-ai_Ornith-1.0-35B-GGUF@f9403b4da6306eb72fde0af1fe2df07cab1f88ce/deepreinforce-ai_Ornith-1.0-35B-Q4_K_M.gguf` |
| llama.cpp, NLL baseline | `bartowski/deepreinforce-ai_Ornith-1.0-35B-GGUF@f9403b4da6306eb72fde0af1fe2df07cab1f88ce/deepreinforce-ai_Ornith-1.0-35B-Q8_0.gguf` |
| mlx-lm and oMLX | `Jundot/Ornith-1.0-35B-oQ4e@1e505ab782d47aeb87a43eebe357d65d8efe9cb7` |

Download each artifact into a stable local location. For the MoEspresso
package, verification is part of acquisition:

```bash
hf download steadfastgaze/Ornith-1.0-35B-Q4_K_M-MoEspresso \
  --revision c67d34262a258f815789a2018341317c755c45a6 \
  --local-dir <ornith-package>

uv run --locked moespresso verify <ornith-package>
```

Before a run, record the package manifest id, manifest hash, model filename,
file size, and SHA-256. Hashing large artifacts can warm the page cache and the
host, so finish it before thermal conditioning.

## Speed protocol

The published speed rows were measured on an M3 Max with 40 GPU cores and
128 GB unified memory. Each table cell follows the same sequence:

Every speed or latency measurement runs with the disk KV tier off
(`MOESPRESSO_DISK_KV=off`), unless the disk KV tier is itself the subject of
the measurement. The tier writes checkpoints on the first request over a
prompt and skips or restores on later ones, so it makes request timing depend
on store state that persists across requests and across processes:
first-request writes inflate TTFT, warm-store runs omit that cost, and any
estimator that compares two requests absorbs the difference into its result.
The published rows were measured with the tier off, before it became the
serving default.

1. Start a fresh engine process and load one model.
2. Run an unmeasured eight-token generation on the selected prompt to compile
   kernels and warm the loaded stack.
3. Discard that logical session or KV cache while leaving the process and
   weights loaded.
4. Start a fresh measured session and generate exactly 256 tokens.
5. Save the prompt token count, first-token timestamp, final-token timestamp,
   raw engine timings, generated-token count, and runtime configuration.
6. Exit the model process.

Run three valid repeats for every cell. Interleave engines by round and rotate
the starting engine one position to the left in each round. The order is
MoEspresso, mlx-lm, llama.cpp, so the second round starts with mlx-lm and the
third with llama.cpp. Report the median; retain all three raw values.

The measured context points are 3,969 prompt tokens for the short cell, 8,191
for the medium cell, and 37,000 for the long cell.

Prepare one complete rendered prompt per context point with the MoEspresso
package tokenizer. Record its numeric token IDs and SHA-256 in the external
evidence directory, then feed the same IDs to every engine that supports
numeric prompt input. If an engine requires text, verify that its tokenizer
reproduces the exact ID sequence before accepting its row. Tokenizing or
rendering during the timed interval invalidates the measurement.

### Matched controls

All arms used:

- greedy selection with temperature 0;
- exactly 256 generated tokens after the eight-token prewarm;
- the same EOG exclusions at the logits for the fixed-length timing boundary;
- thinking disabled;
- vision disabled and a text-only model graph;
- MTP, DFlash, draft models, speculative prefill, and speculative decoding
  disabled;
- prompt reuse, response caches, and disk KV restore disabled;
- one model process and one measured request at a time;
- AC power, Low Power Mode disabled, nominal macOS thermal state, and no
  competing model process or sustained GPU workload.

The mlx-lm run used its stock text graph through an in-process numeric-token
adapter. llama.cpp used `--no-mmproj`.

KV formats were part of each product stack. MoEspresso and mlx-lm used affine
Q8 KV with group size 64. llama.cpp used `q8_0` K and V with group size 32.
These policies quantized the ten full-attention caches; the thirty recurrent
state caches retained their normal representation. Record these choices with
every result.

### Timing and aggregation

Let `N` be the exact prompt-token count, `t0` the start of prompt evaluation,
`t1` the time generated token 1 becomes available, and `t256` the time generated
token 256 becomes available.

```text
TTFT wall                          = t1 - t0
TTFT-derived prompt throughput     = N / (t1 - t0)
steady decode wall                 = t256 - t1
decode throughput                  = 255 / (t256 - t1)
```

The README reports the second and fourth quantities. TTFT includes prompt work
and the first generated-token computation, so its derived rate should not be
interpreted as a pure prefill kernel measurement. Use client-side streamed
token-ready timestamps when an engine runs behind HTTP. For in-process engines,
use the equivalent call boundary and materialized-token timestamps. The
fixed-length in-process adapters perform one final-prompt evaluation and 255
decode evaluations. They do not submit an unused token-257 lookahead call.

Reject and rerun a repeat when any of these conditions occurs:

- the prompt or generated-token count differs;
- the selected sampler, cache, model route, or optional-model path differs;
- a process reuses a measured session or prefix cache;
- another CPU, GPU, memory, or storage workload overlaps the cell;
- the host leaves AC power, enables Low Power Mode, or leaves nominal thermal
  state;
- a crash, truncation, early stop, or missing timing boundary occurs.

Do not numerically correct rejected runs or select the fastest three from a
larger set.

### Temperature and host record

Check thermal state and temperature immediately before and after every cell.
The published session used `mactop` 2.1.5:

```bash
MACTOP_LANG=en mactop \
  --headless --format json --unit-temp celsius --count 1
```

Save the raw output with the cell timestamps. Also record macOS version,
hardware, power state, engine build identity, dependency versions, compiler
flags, and relevant environment variables. The accepted runs began below
56 degrees Celsius and remained at the nominal OS thermal state. Repeat ranges
stayed within 0.66 percent, and no competing model process ran.

### Engine seams

The comparison matrix used the product runtimes with small external adapters
where exact numeric tokens or timing boundaries were unavailable from a public
server response:

- MoEspresso loaded the verified package through its manifest-driven runtime.
  Its in-process adapter recorded exact generated token IDs, token-ready
  timestamps, cache offsets, and runtime engagement counters.
- mlx-lm used the stock text graph and generation semantics through an
  in-process numeric-token adapter. It set `kv_bits=8`, `kv_group_size=64`, and
  `quantized_kv_start=0` explicitly.
- llama.cpp used the pinned `llama-server` with one slot, continuous batching
  disabled, `--no-mmproj`, `--cache-type-k q8_0`, `--cache-type-v q8_0`, and no
  speculative options. The startup log had to confirm both effective cache
  types before timing began.

Keep measurement adapters with the raw benchmark record instead of adding them
to the MoEspresso source or test tree. They apply the recorded controls and
serialize evidence. They must not introduce engine-specific model or sampler
changes.

## Serving protocol for the DeepSeek-V4-Flash release package

No other engine loads the release package
`DeepSeek-V4-Flash-0731-2.37bpw-MoEspressoV2`. Its 3,844-token drafter-on/off
numbers are therefore MoEspresso-only certifications. They follow a different
protocol from the Ornith matrix above, the 37,000-token DeepSeek product-stack
comparison below, and the DeepSeek quality comparison:

- ten fresh-process arms at one 3,844-token prompt, five per side and
  alternating, drafter off first;
- greedy selection at temperature 0, thinking disabled;
- `MOESPRESSO_DISK_KV=off` on every arm;
- `MOESPRESSO_DS4_DRAFTER` pinned per arm rather than left to automatic
  selection, and the resolved state recorded with the arm. Automatic selection
  reads the host's wired budget, so two unpinned arms can differ by more than
  the thing under test;
- a bounded eight-token warmup request ahead of each measured request, with the
  measured request timed on its own single-call boundaries;
- a thermal and headroom check before every arm;
- every arm's generated text checked digest-exact against the package's own
  reference rail, so a timing number that came from a different token stream
  cannot be accepted.

The readings are in [`deepseek_v4_speed.md`](deepseek_v4_speed.md), and the
quality ladder for the same artifact is in
[`deepseek_v4_quality.md`](deepseek_v4_quality.md).

### DeepSeek 37,000-token decode comparison

The release also carries one long-context comparison across three complete
artifact and runtime stacks. It is separate from the ten-arm 3,844-token
drafter comparison:

- the repeated public `long_code_audit.txt` fixture renders to exactly 37,000
  tokens; prompt text, rendered text, and numeric token IDs are hash-pinned;
- MoEspresso sets `MOESPRESSO_DISK_KV=off` and
  `MOESPRESSO_DS4_DRAFTER=off`; DS4 uses its ordinary non-MTP session graph;
  llama.cpp runs with no speculative implementation;
- every fresh process runs an unreported 8-output warmup on the same full
  prompt shape, destroys or erases that logical session, and measures from
  empty state;
- EOG token 1 is excluded at the logits to retain exactly 256 output
  boundaries and 255 after-first timing intervals;
- MoEspresso keeps all 256 experts resident in the capacity-256 pooled graph,
  and route counters must report no request-time expert I/O;
- DS4 uses its native latent cache. llama.cpp uses one slot, disables
  continuous batching and context shift, passes numeric prompt IDs directly,
  sets `cache_prompt=false`, and attests effective `q8_0` K and V cache storage
  from the startup log;
- each process starts only after an exact
  `mactop --headless --count 1` sample reports AC power, Low Power Mode off,
  nominal thermal state, and a GPU reading no higher than 54 C;
- all post samples must remain nominal and on AC;
- every receipt records the 256-token output digest. MoEspresso and llama.cpp
  require repeat-identical digests. DS4 records, but does not require, that
  identity because its low-margin greedy rail varied across otherwise valid
  fresh-process repeats.

The timing boundary and aggregation use the 256-output formulas above. The
published rows report three-run medians and full ranges. MoEspresso and DS4 use
equivalent in-process materialized-token boundaries. llama.cpp uses streamed
client token-ready timestamps; its independently reported server interval must
agree with the client interval. The 3,844-token and 37,000-token records use
different generation shapes, so the documents keep them in separate tables.

This comparison does not isolate runtime speed from quantization or cache
format. MoEspresso uses the 84.35 GB release target, DS4 uses the 86.72 GB
antirez IQ2_XXS GGUF, and llama.cpp uses the 90.86 GB Unsloth UD-IQ2_XXS GGUF.
The artifact and source pins are the same ones listed in the DeepSeek quality
comparison below.

## Quality protocol

### DeepSeek-V4-Flash

The earlier DeepSeek cross-engine comparison uses a private capture of 100
official API continuations, containing 2,290 target tokens. The provider
returned unusable selected-token log probabilities, so the API continuations
serve as a textual oracle. No API perplexity is claimed.

The release package is scored against a later capture of the same 100 prompts
from the checkpoint it was built from, containing 2,313 target tokens, and
against a second capture pass that supplies the provider's own step-level
self-agreement. Those two references hold different continuations, so their scores
are not a before and after. The release ladder and its readings are in
[`deepseek_v4_quality.md`](deepseek_v4_quality.md).

#### DeepSeek-V4-Flash quality comparison

This panel compares three independently packaged DeepSeek-V4-Flash 0731
artifacts. It measures each complete artifact and runtime stack. It does not
isolate an engine from its quantization recipe.

| Artifact and runtime | Target weight files | Target weight bytes | Decimal GB | Drafter treatment |
|---|---:|---:|---:|---|
| MoEspresso 2.37 bpw / MoEspresso | 47 safetensors | 84,354,585,696 | 84.3546 | 6,388,822,111-byte DSpark sidecar excluded |
| antirez IQ2_XXS / DS4 | 1 GGUF | 86,720,111,488 | 86.7201 | no drafter payload |
| Unsloth UD-IQ2_XXS / llama.cpp | 3 GGUF shards | 90,860,736,928 | 90.8607 | no drafter payload |

The sizes are logical model-file sizes in decimal GB. They are not complete
repository download sizes. Small package furniture is outside the MoEspresso
weight count, while GGUF headers remain part of their files.

| Package and runtime | WikiText mean NLL | WikiText PPL | API-continuation mean NLL | API selected-token agreement | API first-token agreement |
|---|---:|---:|---:|---:|---:|
| MoEspresso 2.37 bpw / MoEspresso | 1.714662117 | 5.554798326 | 0.396341911 | 2022/2313 (87.42%) | 66/100 |
| antirez IQ2_XXS / DS4 | 1.768240384 | 5.860532000 | 0.415167895 | 1996/2313 (86.29%) | 54/100 |
| Unsloth UD-IQ2_XXS / llama.cpp | 1.716948 | 5.5675 | 0.362239186 | 2052/2313 (88.72%) | 62/100 |

##### WikiText controls

The corpus is the held-out test split of `Salesforce/wikitext`,
`wikitext-2-raw-v1`, distributed as `wiki.test.raw`: 1,290,590 bytes with
SHA-256
`173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`.
The panel takes the first 32 complete, contiguous, non-overlapping 2,048-token
windows. Each window starts with fresh state. Positions 0 through 1,024 provide
context, and positions 1,025 through 2,047 are the 1,023 scored targets. The
aggregate therefore contains 32,736 target losses.

All three engines produced identical token IDs over the 65,536-token benchmark
prefix. Its little-endian uint32 SHA-256 is
`641ecd4ce1149af6e90cc66837daf0fd6f3a14bba1c006ba35a4cbe6b31edd2c`.
For every target, the scorer takes the negative log probability under the full
output distribution. Aggregate target loss is divided by 32,736 for mean NLL,
and `exp(mean NLL)` is reported as perplexity. NLL and perplexity are the same
measurement in different units.

MoEspresso uses a direct full-model forward with the routed pool at full
capacity. The DSpark component can be resolved during package load, but no
speculative generation call occurs. Prefix caches and disk KV do not
participate. DS4 uses the same prompt and target files through its quality
scorer. llama.cpp runs with a 2,048-token context, logical batch 2,048, and
physical microbatch 512. Changing llama.cpp's physical grouping changed the
cumulative loss slightly, so the grouping is part of the recorded protocol.

This last-half panel is not the all-position MoEspresso release gate. The gate
scores 65,504 targets over the same number and size of windows and reports
6.4774 for this package. The matched comparison reports 5.5548 over 32,736
targets. Neither value supersedes the other.

##### Calibration boundary

MoEspresso was not calibrated on this WikiText test panel. The package's
routed-expert importance statistics and allocation objective use a separate
mixed training spine: the pinned Bartowski `calibration_datav5` corpus joined
with exllamav3's `technical.utf8`. Neither the WikiText-2 test split nor the
known historical train-split excerpt emitted importance-matrix data, moments,
or optimizer inputs. WikiText was used later as held-out quality evidence.

The disjointness audit found zero shared exact lines of 48 characters or more
between either calibration component and the test split. A separate full-file
comparison against Bartowski v5 found no substantive exact overlap with the
test split or the known historical train excerpt; no normalized shared run was
longer than eight case-folded, whitespace-normalized tokens. This supports the
panel-specific calibration claim. It does not prove that the calibration
corpus contains no Wikipedia-derived prose, because its complete upstream
source genealogy is not established.

##### API-continuation controls

The API panel uses one pinned official-provider pass over 100 prompts, with
thinking disabled and temperature zero. It contains 2,313 provider-selected
continuation tokens. Each local engine receives the same rendered prefix and
is teacher-forced under that continuation. Mean NLL is the local negative log
probability of the provider-selected token. Selected-token agreement compares
the local argmax with the provider-selected token at each teacher-forced step;
first-token agreement applies the same comparison to the first step of each
case.

The provider row compares selected tokens from a second independent pass over
the same rail. It is a run-to-run reproducibility reference, not a mathematical
upper bound. The provider's selected-token log probabilities were unusable, so
the provider row has no NLL. The hosted API does not expose the arbitrary-corpus
full logits needed for a WikiText NLL or perplexity arm. These columns also do
not measure free-running generation agreement because every local step is
conditioned on the provider continuation.

##### Artifact and implementation pins

- MoEspresso revision
  `b1444abeed0ae5f031a26a8ace07060d0edc7c85`; package artifact id
  `pkg:d46e752414eaa44d4d5d661e7700feef69dacc9c12548206035f32128ce36bdf`;
  package-manifest SHA-256
  `01e1b8863aafc7476d823307e6f4f70911efcad8f76932eb098af073bedc7b01`;
  repository `steadfastgaze/DeepSeek-V4-Flash-0731-2.37bpw-MoEspressoV2`.
- DS4 revision `b0309611041655f4e45671cfd9c9886aff161406`; artifact
  `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`;
  GGUF SHA-256
  `ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0`.
- llama.cpp revision `936918514ce522b553c0fd80b169a6440e6096c6`; Unsloth repository
  `unsloth/DeepSeek-V4-Flash-0731-GGUF`, variant `UD-IQ2_XXS`, snapshot
  `fbbb5b93fb787c21338159b0af3318bb3f4d9768`.
- Unsloth shards
  `DeepSeek-V4-Flash-0731-UD-IQ2_XXS-00001-of-00003.gguf`,
  `DeepSeek-V4-Flash-0731-UD-IQ2_XXS-00002-of-00003.gguf`, and
  `DeepSeek-V4-Flash-0731-UD-IQ2_XXS-00003-of-00003.gguf`; SHA-256 values in
  the same order:
  `c58c9d62eac7b62e9578b52613f425e48313d7212ab8d1d76caed8ea8de26595`,
  `65a113df6d4469f16db6882b6919e153c464c3c78c833f5e1b41a33803cdbd52`,
  and `a69102ddfaf4a84426e11fdb66716654f4260dc3a1de3ade9fd50e006b8691d3`.

The two GGUF payloads each declare 1,328 tensors. Their complete tensor-name
inventories are identical: target blocks `blk.0` through `blk.42` plus six
global target tensors. Neither contains an MTP block or drafter tensor. The
source checkpoint contains 4,705 explicit tensors under `mtp.0`, `mtp.1`, and
`mtp.2`, totaling 10.86 GB before conversion; none occurs in either measured
GGUF. The DS4 GGUF retains `deepseek4.nextn_predict_layers = 1` as metadata,
but there is no corresponding weight block.

Holders of the private capture can run MoEspresso Q0, Q2, and Q3 with explicit
paths and retain each JSON report:

```bash
uv run --locked moespresso-ds4-quality \
  q0 --package <deepseek-package> --json-out <q0.json>

uv run --locked moespresso-ds4-quality \
  q2 --package <deepseek-package> \
  --reference <private-reference.json> --json-out <q2.json>

uv run --locked moespresso-ds4-quality \
  q3 --package <deepseek-package> --json-out <q3.json>
```

For the earlier 2,290-token capture, every engine computes full-vocabulary
log-softmax at each target position and retains the negative log probability of
the reference token. Aggregate all 2,290 values with `math.fsum`, divide by
2,290 for mean NLL, and exponentiate that mean for perplexity. Q3 uses the same
deterministic story, thinking-off render, greedy selection, and 256-token cap in
every engine it runs in.

That earlier capture is validated by scoring the same rendered prompts and
exact continuation texts through independent reference implementations
alongside MoEspresso, so a result cannot rest on one scorer. Its per-engine
readings are not published here.

The upstream public official-vector gate is four vectors and 13 next-token
decisions, and its suite excludes `long_memory_archive` because the captured API
vector and official graph disagree. This is why the README labels that cell as a
native token gate rather than MoEspresso Q0.

The private capture, local target IDs, per-token losses, continuations, and
per-case records stay outside the repository. Public evidence may contain the
aggregate values, case counts, total target count, engine and artifact
identities, and stated limitations. A reader without the private capture can
reproduce Q0, Q1, and Q3, or create a new independent Q2 reference as described
in [`deepseek_v4_quality.md`](deepseek_v4_quality.md). A new 100-case reference
requires 100 API requests and produces a different comparison set.

### Ornith

The Ornith NLL matrix uses the WikiText-2 raw v1 test split from the public
`ggml-org/ci` mirror at revision
`927b3642933080f1b0e811e2f916e14c292992f9`. Verify these identities:

- archive `wikitext-2-raw-v1.zip`, SHA-256
  `ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11`;
- member `wikitext-2-raw/wiki.test.raw`, 1,290,590 bytes, SHA-256
  `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`.

Tokenize with the verified MoEspresso Ornith tokenizer and
`fix_mistral_regex=True`. Select twelve sequential, non-overlapping windows of
1,024 tokens starting at corpus token zero. No state carries between windows.
The MoEspresso, mlx-lm, and oMLX arms score each complete window without a
generation cache. The llama.cpp scorers start a fresh F16 KV context for every
window. None of these NLL arms evaluates Q8 KV. Score logits positions 512
through 1,022 against target tokens 513 through 1,023. This produces 511 scores
per window and 6,132 scores in total.

For every target, normalize over all 248,320 output logits. Convert each
negative log probability to float64, aggregate with `math.fsum`, and compute:

```text
mean_nll  = math.fsum(token_nlls) / 6132
perplexity = math.exp(mean_nll)
delta_nll = candidate_mean_nll - gguf_q8_mean_nll
```

Run those exact windows through GGUF Q8_0, GGUF Q4_K_M, the MoEspresso Q4_K_M
package, oMLX oQ4e, and the byte-identical oQ4e artifact through stock mlx-lm.
Save all twelve window means, the 6,132 finite target losses, token and corpus
hashes, full artifact identities, scorer build identity, and raw output. GGUF
Q8_0 is the high-fidelity quantized baseline. The mlx-lm oQ4e arm measures
perplexity 6.2897 versus 6.2855 through oMLX. This descriptive comparison does
not establish an equivalence threshold.

The mlx-lm teacher-forced scorer does not use a generation KV cache. Its NLL
validates aggregate weight and graph behavior. Qualify its Q8 timing arm with a
separate natural-stop generation. At 8,191 tokens the affine-Q8 arm selected EOG
after a short continuation, and a matched raw-BF16 arm selected a different
first token. This is prompt-specific cache-sensitivity evidence, so the mlx-lm
Q8 row remains timing evidence without a cache-quality equivalence claim.

Run the served acceptance gate separately:

```bash
uv run --locked moespresso-ornith-gate \
  <ornith-package> --json-out <ornith-gate.json>
```

Its private reasoning questions and answers must remain in the ignored fixture
tree. The public coding and long-context families can be run independently as
described in [`ornith_quality.md`](ornith_quality.md).

## Evidence checklist

Keep the following together for each published table revision:

- complete engine and dependency versions;
- clean source revisions and build flags;
- artifact repository revisions, filenames, sizes, and SHA-256 values;
- package manifest ids and hashes;
- prompt or corpus identity, exact token counts, and token-array hashes;
- sampling, EOG exclusion, thinking, vision, speculation, MTP/DFlash, and KV
  settings;
- adapter source hashes and the exact model-call count;
- effective KV attestation from runtime counters or startup logs;
- three raw speed repeats and the interleaved run order;
- prewarm, process, cache, and timing-boundary records;
- thermal, power, OS, and hardware records;
- quality aggregate inputs and outputs within their public/private boundary.

Family-specific speed records are in
[`deepseek_v4_speed.md`](deepseek_v4_speed.md) and
[`ornith_speed.md`](ornith_speed.md).
