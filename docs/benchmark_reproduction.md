# Published comparison protocol

The README tables compare complete engine, artifact, and cache stacks. This
document defines the controls and evidence required for an independent
repetition. MoEspresso does not ship a cross-project benchmark framework,
copies of third-party runners, exact prompt-token fixtures, or benchmark-only
measurement adapters.

Keep raw results and any temporary adapters in a separate work directory. Large
downloads happen before the measurement session and run one at a time.

## Versions and artifacts

The published matrix used these engine versions:

| Engine | Version |
|---|---|
| MoEspresso | `1.0.0` |
| mlx-kquant | `0.3.0` at `e165cafafa149493d298871c610e29e95ffa8f10` |
| mlx-lm | `0.31.3` with MLX `0.31.2`, source `15b522f593b7ca5fbc0cac6f7572d40859d2d8fe` |
| oMLX | `0.5.1` |
| DwarfStar | `80ebbc396aee40eedc1d829222f3362d10fa4c6c` |
| llama.cpp | `6eddde06a4f25d55d538b5d15628dcc2b6882147` |

The mlx-kquant revision is the public commit whose source tree matches the
benchmark build.

The model inputs were pinned to immutable Hugging Face revisions:

| Family and engine | Artifact |
|---|---|
| DeepSeek, MoEspresso | `steadfastgaze/DeepSeek-V4-Flash-IQ2_XXS-MoEspresso@777b4488782afcb52327197aee13df9dddd31249` |
| DeepSeek, DwarfStar and llama.cpp | `antirez/deepseek-v4-gguf@9170bf42beb77f38006e016503ecace31f2bd9a0/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf` |
| DeepSeek, oMLX | `Jundot/DeepSeek-V4-Flash-oQ2.5e-mtp@11eb2531c878e6c58405b7ddedb6909d64203ee2` |
| Ornith, MoEspresso | `steadfastgaze/Ornith-1.0-35B-Q4_K_M-MoEspresso@c67d34262a258f815789a2018341317c755c45a6` |
| Ornith, llama.cpp | `bartowski/deepreinforce-ai_Ornith-1.0-35B-GGUF@f9403b4da6306eb72fde0af1fe2df07cab1f88ce/deepreinforce-ai_Ornith-1.0-35B-Q4_K_M.gguf` |
| Ornith NLL baseline, llama.cpp | `bartowski/deepreinforce-ai_Ornith-1.0-35B-GGUF@f9403b4da6306eb72fde0af1fe2df07cab1f88ce/deepreinforce-ai_Ornith-1.0-35B-Q8_0.gguf` |
| Ornith, mlx-lm and oMLX | `Jundot/Ornith-1.0-35B-oQ4e@1e505ab782d47aeb87a43eebe357d65d8efe9cb7` |

Download each artifact into a stable local location. For the two MoEspresso
packages, verification is part of acquisition:

```bash
hf download steadfastgaze/DeepSeek-V4-Flash-IQ2_XXS-MoEspresso \
  --revision 777b4488782afcb52327197aee13df9dddd31249 \
  --local-dir <deepseek-package>

hf download steadfastgaze/Ornith-1.0-35B-Q4_K_M-MoEspresso \
  --revision c67d34262a258f815789a2018341317c755c45a6 \
  --local-dir <ornith-package>

uv run --locked moespresso-verify <deepseek-package>
uv run --locked moespresso-verify <ornith-package>
```

Before a run, record the package manifest id, manifest hash, model filename,
file size, and SHA-256. Hashing large artifacts can warm the page cache and the
host, so finish it before thermal conditioning.

## Speed protocol

The published speed rows were measured on an M3 Max with 40 GPU cores and
128 GB unified memory. Each table cell follows the same sequence:

The published rows were measured with the disk KV tier off, before it became
the serving default; disable it for a like-for-like run
(`MOESPRESSO_DISK_KV=off moespresso-serve <package>`), or first-request
checkpoint writes and later restores skew TTFT in both directions.

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
the starting engine one position to the left in each round. For example, a
DeepSeek order of MoEspresso, DwarfStar, llama.cpp, oMLX starts the second round
with DwarfStar and the third with llama.cpp. The Ornith order is MoEspresso,
mlx-lm, llama.cpp, followed by the same left rotation. Report the median; retain
all three raw values.

The measured context points are:

| Family | Short | Medium | Long |
|---|---:|---:|---:|
| DeepSeek-V4-Flash | 3,844 | 7,698 | 15,406 |
| Ornith | 3,969 | 8,191 | 37,000 |

Prepare one complete rendered prompt per family and context point with the
MoEspresso package tokenizer. Record its numeric token IDs and SHA-256 in the
external evidence directory, then feed the same IDs to every engine that
supports numeric prompt input. If an engine requires text, verify that its
tokenizer reproduces the exact ID sequence before accepting its row. Tokenizing
or rendering during the timed interval invalidates the measurement.

### Matched controls

All arms used:

- greedy selection with temperature 0;
- exactly 256 generated tokens after the eight-token prewarm;
- the same family-specific EOG exclusions at the logits for the fixed-length
  timing boundary;
- thinking disabled;
- vision disabled and a text-only model graph;
- MTP, DFlash, draft models, speculative prefill, and speculative decoding
  disabled;
- prompt reuse, response caches, and disk KV restore disabled;
- one model process and one measured request at a time;
- AC power, Low Power Mode disabled, nominal macOS thermal state, and no
  competing model process or sustained GPU workload.

The oMLX DeepSeek repository contains MTP weights, but the run did not load or
execute them. The Ornith mlx-lm run used its stock text graph through an
in-process numeric-token adapter. llama.cpp used `--no-mmproj`. DwarfStar used
its ordinary non-MTP graph.

KV formats were part of each product stack. Ornith used affine Q8 KV with group
size 64 for MoEspresso and mlx-lm. llama.cpp used `q8_0` K and V with group size
32. These policies quantized the ten full-attention caches; the thirty recurrent
state caches retained their normal representation. DeepSeek used each engine's
native latent-cache representation. Record these choices with every result.

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
- DwarfStar used its public evaluation/session API on the pinned DeepSeek GGUF.
- oMLX used version 0.5.1 and its `BatchedEngine` text path for DeepSeek, with
  paged SSD cache, TurboQuant KV, prefix cache, vision, and speculative features
  disabled.

Keep measurement adapters with the raw benchmark record instead of adding them
to the MoEspresso source or test tree. They apply the recorded controls and
serialize evidence. They must not introduce engine-specific model or sampler
changes.

## Quality protocol

### DeepSeek-V4-Flash

The comparison uses a private capture of 100 official API continuations,
containing 2,290 target tokens. The provider returned unusable selected-token
log probabilities, so the API continuations serve as a textual oracle. No API
perplexity is claimed.

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

Score the same rendered prompts and exact continuation texts with the pinned
DwarfStar GGUF and the pinned oMLX artifact. For every target position, compute
full-vocabulary log-softmax and retain the negative log probability of the
reference token. Aggregate all 2,290 values with `math.fsum`, divide by 2,290
for mean NLL, and exponentiate that mean for perplexity. Q3 uses the same
deterministic story, thinking-off render, greedy selection, and 256-token cap in
all three engines.

DwarfStar also runs its upstream public official-vector gate: four vectors and
13 next-token decisions. Its upstream suite excludes `long_memory_archive`
because the captured API vector and official graph disagree. This is why the
README labels that cell as a native token gate rather than MoEspresso Q0.

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
