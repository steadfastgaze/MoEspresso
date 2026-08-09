# Published comparison protocol

The README tables compare complete engine, artifact, and cache stacks. This
document defines the controls and evidence required for an independent
repetition. The cross-engine sections that follow, from artifact pinning
through engine seams, cover Ornith. The DeepSeek-V4-Flash release package is
served and scored by MoEspresso alone, under the separate serving protocol and
quality protocol further down. MoEspresso does not ship a cross-project
benchmark framework, copies of third-party runners, exact prompt-token
fixtures, or benchmark-only measurement adapters.

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

uv run --locked moespresso-verify <ornith-package>
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

No cross-engine matrix has been measured on the release package
`DeepSeek-V4-Flash-0731-2.37bpw-MoEspressoV2`: no other engine loads a
MoEspresso package, and the public third-party DeepSeek artifacts are built from
a different checkpoint. Its published serving numbers are MoEspresso-only
certifications, and they follow a different protocol from the cross-engine
matrix above:

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

## Quality protocol

### DeepSeek-V4-Flash

The earlier DeepSeek cross-engine comparison uses a private capture of 100
official API continuations, containing 2,290 target tokens. The provider
returned unusable selected-token log probabilities, so the API continuations
serve as a textual oracle. No API perplexity is claimed.

The release package is scored against a later capture of the same 100 prompts
from the checkpoint it was built from, containing 2,313 target tokens, and
against a second capture pass that supplies the provider's own step-level
self-noise. Those two references hold different continuations, so their scores
are not a before and after. The release ladder and its readings are in
[`deepseek_v4_quality.md`](deepseek_v4_quality.md).

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

The scoring method, for any engine scoring these fixtures: for every target
position, compute full-vocabulary log-softmax and retain the negative log
probability of the reference token. Aggregate all 2,290 values with `math.fsum`,
divide by 2,290 for mean NLL, and exponentiate that mean for perplexity. Q3 uses
the same deterministic story, thinking-off render, greedy selection, and
256-token cap in every engine it runs in.

The private capture is validated by scoring the same rendered prompts and exact
continuation texts through independent reference implementations alongside
MoEspresso, so a result cannot rest on one scorer. Those readings are not
published.

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
