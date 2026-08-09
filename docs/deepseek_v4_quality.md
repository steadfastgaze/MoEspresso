# DeepSeek-V4-Flash quality gates

DeepSeek-V4-Flash has five numbered model-specific quality gates plus a
natural-text perplexity gate. They are manual real package runs: they do not run
in `make test`, they do not run in CI, and they never discover a package
implicitly.

The gates answer different questions. Passing a higher-numbered gate does not
replace the lower gates, and package verification does not replace any of them.

| Gate | Question | Model forward | Bar |
|---|---|---|---|
| Q0 | does request text reach the expected token rail? | no | coded |
| Q1 | does greedy decoding select the reference's tokens? | yes | coded |
| Q2 | what likelihood does the package assign the reference continuations? | yes | package dependent, reported |
| Q3 | does long-context recall survive at depth? | yes | coded |
| Q4 | does the distribution behind the tokens match a local teacher? | dumps | instrument checks coded, levels reported |
| WikiText PPL | does the package stay numerically sane on ordinary prose? | yes | declared per run |

## Run the gates

Pass `--package` explicitly or set `MOESPRESSO_DS4_QUALITY_PACKAGE`:

```bash
uv run --locked moespresso-ds4-quality \
  q0 --package <package-dir>

uv run --locked moespresso-ds4-quality \
  q1 --package <package-dir> [--vectors-root <reference-dir>]

uv run --locked moespresso-ds4-quality \
  q2 --package <package-dir> --reference <private-reference.json>

uv run --locked moespresso-ds4-quality \
  q3 --package <package-dir>

uv run --locked moespresso-ds4-wikitext-ppl \
  --package <package-dir> --corpus <corpus.txt> --limit <perplexity-bar>

uv run --locked moespresso-ds4-q4 \
  --teacher <teacher.npz> --candidate <candidate.npz> --markers <markers.json>
```

Every gate accepts `--json-out <path>`. Each run records the installed MLX wheel
tag because different wheel builds of one MLX version can form different,
internally deterministic numeric lattices.

The gate runner pins all experts resident before model work unless an explicit
environment override is already set. This makes evidence comparable: cold and
fully resident routed prefill use valid but differently ordered reductions that
can flip knife-edge tokens.

## Q0: renderer and tokenizer goldens

Q0 performs no model forward. It checks the packaged tokenizer and the
DeepSeek prompt renderer against committed goldens copied from the public ds4
reference test-vector suite.

Q0 catches tokenizer identity, control-token, and thinking-off rendering drift.
It proves that request text reaches the expected token rail; it does not prove
model math.

## Q1: greedy selected-token identity

Q1 serves five committed public prompts with greedy decoding, temperature 0,
thinking disabled, and a local top-20 capture, then asks one question of every
step the reference holds: is the reference's selected token the local rank-0
token?

For every comparable step, the contract requires:

- the reference selected token to be the local rank-0 token;
- the expected prompt, render, model, and run metadata.

**The step total comes from the reference fixture.** It is not a constant. The
same five prompts produce different step totals on different checkpoints,
because a checkpoint that selects an end-of-sequence token earlier records
fewer steps at the same `max_tokens`. The gate cross-checks the manifest's
declared count against the number of steps it scored, so a shorter run is a
finding rather than a quieter pass.

**Comparability is a property of the reference.** A reference step whose
selected token does not resolve to exactly one token under the package
tokenizer cannot be scored as identity, and Q0 already pins that every reference
token does resolve, so a non-comparable step fails closed. A candidate that
stops generating before the reference runs out of steps fails the remaining
steps instead of skipping them: the bar cannot be lowered by generating less.

**The top-20 overlap clause is retired.** The provider stack accepts
`top_logprobs: 20` and returns sentinel values: the selected entry carries a
real number and every other candidate degrades to token-id-ordered filler at
`-9999`. An overlap of at least one is therefore satisfied by the selected token
alone and measures nothing. The candidate and reference top-20 id sets and their
overlap are still recorded in the evidence, because that is what the fixture
holds, but no threshold reads them. Calibrated distribution questions belong to
Q4.

**Reference identity.** The gate accepts three names for a reference of record:
the undated model name the published upstream suite carries, the dated slug a
self-capture sends, and the canonical dated id a provider generation record
reads back. They are naming conventions, not different contracts. An undated
slug is not a checkpoint identity and is never used to capture a reference.

**Prompts and reference records have separate roots.** `--fixture-root` holds
the public prompt files; `--vectors-root` holds the reference manifest and its
official records and defaults to the fixture root. A self-captured reference is
oracle material that lives outside the committed tree while the prompts it was
captured on stay public, so scoring reads the same public prompt bytes against a
private reference.

Q1 counts are keyed to the MLX wheel lattice. One package scored against one
reference can land a step apart on two lattices, separated by a single
deterministic casing knife edge and reproducible in repeated runs on each. A Q1
count quoted without its lattice is not a comparable number.

Q1 catches prompt drift and serving errors large enough to move short and
medium continuations. It samples the distribution and does not prove full-logit
identity.

## Q2: target-token negative log likelihood

Q2 teacher-forces provider continuations over the tracked 100-prompt set and
measures the negative log likelihood assigned by the local package. It is the
continuous comparison instrument for kernel numerics, package realizations,
and low-bit recipe changes.

Create a new reference with:

```bash
uv run --locked moespresso-ds4-quality \
  q2-capture [--out <private-reference.json>] [--force]
```

The capture command reads `OPENROUTER_TOKEN` from the environment or `.env` and
does not write the token into the payload. The resulting continuations and
top-logprob records are API-derived oracle material and must remain outside the
committed public tree.

**A capture never lands on an existing reference.** Without `--out` it writes a
new timestamped file under the private capture root, and any output path that
already exists is refused before the first request unless `--force` says
otherwise. Promoting a capture to the reference of record is a separate,
deliberate act. The model default is the dated slug: an undated slug is a
distinct model entry whose published canonical id can name an earlier build, so
a capture sent on it cannot be attributed to a checkpoint from the response
alone.

**Which reference `q2` scores against.** `--reference` defaults to the reference
of record. An earlier reference stays readable so historical scores remain
reproducible, and it is passed explicitly. The two hold different checkpoints'
continuations, so their numbers are not comparable and must not be presented as
a before and after. A second independent capture pass over the same prompts sits
beside the reference of record as the self-noise witness: it says how much of
any difference is the provider rather than the package.

Compare two Q2 score artifacts with:

```bash
uv run --locked moespresso-ds4-quality \
  q2-compare <old-q2.json> <new-q2.json>
```

Q2 reports aggregate and per-case NLL plus case wins. It has no universal
package-independent threshold; compare packages on the same prompt and reference
set and MLX wheel. Every anchor value belongs to the reference and package that
produced it, and none of them transfers across a reference change.

## Q3: deterministic long-context fact recall

Q3 generates a deterministic long story with 16 embedded name-to-number facts
and requires exact `Name=number` answer lines. The fixture generator, manifest,
and answers are public and API-free.

Q3 exercises the compressed-attention, indexer, long-prefill, and cache paths
that the shorter Q1 prompts barely touch.

## Q4: the teacher-forced KL panel

Q0 through Q3 judge tokens. Q4 judges the distribution behind them. It scores a
candidate's teacher-forced logits against a locally streamed teacher over
identical positions, and it loads no model of its own: both sides arrive as
dumps, so the panel is cheap to re-score and easy to keep honest.

The teacher is local, not a provider. The hosted stack returns sentinel values
for every non-selected candidate, so a calibrated reference distribution can
only come from a teacher-forced pass over the unquantized weights on this host.

Per probe the panel reports:

- **mean KL** over the teacher's top-K support, both sides renormalized inside
  that support. Renormalizing makes it a divergence between two distributions on
  one support, so it cannot be negative and a negative value is a defect rather
  than a reading;
- **top-1 agreement**, over all scored positions and over the conditioned subset;
- **entropy-conditioned KL**, restricted to positions in the top quintile of
  teacher entropy. The threshold comes from the teacher alone, so two candidate
  arms are conditioned on identical positions and the mask cannot move under the
  thing being measured;
- **marker-mass inflation** on overthinking markers: per surface form, as an
  aggregate, and as a narrow subset restricted to the hesitation and
  branch-alternative classes. The narrow column exists because the marker set
  does not survive tokenization uniformly; several candidate surface forms have
  no single-token representation in this vocabulary, so the aggregate mixes
  classes with very different coverage. Marker mass is deliberately not
  renormalized, because renormalizing hides the quantity of interest: how much
  probability the candidate spends on these tokens. Only markers inside the
  teacher's top-K support are observable, and the panel says so;
- **free-run length accounting**, supplied by the caller. Length is the symptom
  the KL columns cannot see: a package can hold its distribution and still run
  long, and a package can shorten because it terminates early rather than
  because it answers concisely.

**Dump schema** (`ds4-q4-teacher-dump-v1`): an npz with `ids` (C, T), `top_ids`
(C, T, K), `top_logits` (C, T, K), `log_partition` (C, T), and `argmax` (C, T).
It adapts an earlier teacher-match format that stored log-probabilities already
renormalized inside the top-K support; storing raw logits with the full-vocab
log-partition instead makes the top-K mass a measured quantity, so a probe whose
teacher distribution is not covered by K is visible in the panel rather than
silently assumed away. The earlier field name still reads, and the evidence
labels which normalization produced it. The candidate dump mirrors the teacher's
with `top_logits` gathered at the teacher's `top_ids`, its own `log_partition`,
and its own `argmax`.

**What gates.** Only properties of the instrument: finite and non-negative KL,
agreement inside `[0, 1]`, teacher and candidate covering the same positions, a
top-K mass in `(0, 1]`, finite marker ratios. Every level is package dependent,
lands in structured output for ledger comparison, and gates only against bars
the caller declares with `--kl-mean-max`, `--top1-agreement-min`, and
`--marker-ratio-max`.

The marker set is passed with `--markers` or `MOESPRESSO_DS4_Q4_MARKERS` and has
no default location. It is campaign material that resolves overthinking surface
forms against the model tokenizer, and it is not distributed with the package.

## WikiText teacher-forced perplexity

The natural-text arm of the acceptance bar, and the reason it is a bar at all:
an activation-rotation overflow once produced a non-finite perplexity here while
Q0 through Q3 and every serve probe stayed finite and the code-corpus arm scored
normally. Overflow hides on ordinary prose.

```bash
uv run --locked moespresso-ds4-wikitext-ppl \
  --package <package-dir> \
  --corpus <corpus.txt> \
  --limit <perplexity-bar> \
  [--corpus-sha256 <digest>] [--window-size 2048] [--window-count 32] \
  [--json-out <path>]
```

The protocol is the first complete, contiguous, non-overlapping windows of a
digest-pinned corpus; one independent direct forward per window with no cache,
so no window can carry state into the next; float32 logits; per-position loss
`logsumexp(logits) - target_logit`; an MLX sum per window and a Python float sum
across windows. Per-window scores stay separate from the aggregate, so a single
non-finite window stays visible instead of being absorbed.

**The corpus is configured, never vendored.** It is third-party text. Pass
`--corpus` or set `MOESPRESSO_DS4_WIKITEXT_CORPUS`. When it is absent the gate
fails before any model work with a message naming the expected file, its
recorded sha256, and its byte count. A digest that does not match is refused
rather than reported, because scores on different corpus bytes are not
comparable; `--corpus-sha256` runs a different corpus deliberately. When the
corpus is the recorded one, its token count under the package tokenizer is
checked, which catches a package whose tokenizer is not the one this protocol
was recorded on.

The corpus of record is the held-out `test` split of `Salesforce/wikitext`,
`wikitext-2-raw-v1`, distributed as `wiki.test.raw` (1,290,590 bytes, sha256
`173c87a5…`, 287,730 tokens under the DeepSeek-V4-Flash tokenizer). The digest
is the public dataset file's, so the pin is checkable by anyone who downloads
the split. An earlier pin named a file that a third-party converter ships as
calibration data, whose own notice describes it as a contiguous subset of the
**train** split; a held-out bar scored on training text measures the wrong
thing. Every recorded perplexity run already used the test split, so this
changed no recorded number; the tracked default caught up with the practiced
protocol.

The default window count is 32, the count every recorded package limit was
measured at. `--window-count` still overrides; the older 8-window readings are
comparable only against other 8-window readings.

**The limit is declared, never assumed.** Perplexity is package-family
dependent: two packages of the same weights read differently on the same corpus,
and a new checkpoint moves both. `--limit` is required and the evidence records
which bar produced the status.

## Release package scorecard

The release package `DeepSeek-V4-Flash-0731-2.37bpw-MoEspressoV2` (recipe in
[`deepseek_v4_package_recipe.md`](deepseek_v4_package_recipe.md)) scores the
full ladder against the DeepSeek-V4-Flash 0731 release. One MLX wheel lattice,
experts pinned resident by the gate runner, `MOESPRESSO_DS4_DRAFTER=off` so the
bundled drafter cannot engage, and every reading against the reference captured
from the checkpoint the package was built from:

| Gate | Reading |
|---|---|
| Q0 renderer and tokenizer goldens | 34/34 |
| Q1 greedy selected-token identity | 9 of 14 |
| Q2 average NLL, 100 prompts and 2,313 target tokens | 0.3963 |
| Q2 step-level selected-token agreement | 87.42 percent |
| Q2 first-token matches | 66/100 |
| Q3 long-context fact recall | 16/16 |
| Q4 served KL panels | valid, 0 findings |
| WikiText test-split perplexity, 32 windows | 6.4774 |

How to read them:

- **Q1.** One prompt diverges at its first greedy token, which costs the four
  steps of that trajectory, and one further step is a single-token knife edge.
  The remaining steps are exact. Q1 counts are keyed to the MLX wheel lattice
  and to the reference; a count quoted without both is not comparable.
- **Q2.** The reference is a provider capture, so the agreement column needs
  the provider's own noise floor beside it: two capture passes over the same
  prompts agree with each other on 89.67 percent of steps, and the package
  reaches 87.42 percent on the same rail.
- **WikiText.** The perplexity reading pairs with its teacher. A bf16
  teacher-forced pass over the same 32 windows scores 4.8886.

The pooled IQ_K runtime has an additional graph-regression gate on the public
package. A fresh capacity-256 forward over one 2,048-token Q4 row matched the
archived resident-graph oracle at raw word precision: 131,072 stored-support
logits, 2,048 log partitions, 2,047 target losses, and 2,048 argmax ids all had
zero mismatches. All 43 routed layers exercised the pooled sorted-prefill path,
with no expert loads, evictions, or bundle reads. The route recorded zero
projection-load waits, index-sync calls, and index-resync calls. The oracle
payload remains private; the counts state the scope of the comparison without
publishing its token or logit contents.

Serving speed for this package is in
[`deepseek_v4_speed.md`](deepseek_v4_speed.md); the acquisition and measurement
controls are in [`benchmark_reproduction.md`](benchmark_reproduction.md).

## Speculative decoding correctness

Speculative decoding
([`speculative_decoding.md`](speculative_decoding.md)) changes when forwards
run, not what is accepted:

- **Lossless acceptance rule.** With temperature 0 a draft token is accepted
  only when it equals the target argmax at its position, computed on the
  verify forward. With temperature above 0 the standard rule (accept with
  probability `min(1, p_target/p_draft)`, resample the first rejection from
  the normalized residual distribution) recovers the target distribution
  exactly; the implementation is checked by a 20,000-trial statistical
  recovery test.
- **Bit-exact verify rollback.** Rejected positions are removed by restoring
  a pre-verify snapshot of the bounded mutable cache state, not by trimming.
  Trimming cannot be exact on the hybrid caches: it clears partial-window
  pool buffers whose source tokens are never re-fed, and the rotating local
  window stops being trimmable once it wraps, at which point trimming
  silently removes nothing. The rollback contract is pinned by tests: a full
  restore is bitwise identical, and the restored state plus its continuation
  logits are bitwise independent of the rejected suffix across pre-wrap,
  post-wrap, and compression-boundary regimes.
- **The verify lattice.** The multi-token verify forward and the fused
  single-token decode forward are distinct numeric lattices, in the same
  class as the documented MLX-wheel and cold/warm residency lattices: valid
  but differently ordered floating-point reductions. Measured on the real
  package, block-row and stepwise logits differ by 0.25-0.94 logit units
  while the argmax holds at gaps of 1.0-3.4. At knife-edge margins the
  speculative and plain token streams can therefore diverge; recorded first
  divergences on the standard prompts sit between tokens 23 and 54, with
  both streams coherent.

Bitwise identity between the speculative stream and the fused single-token
stream is not a general property without giving up the multi-token verify
forward. The acceptance evidence for speculative serving is therefore: the
rollback bit-exactness above, the speculative stream being an exact greedy
decode under the verify lattice, the quality gates passing with speculation
enabled, and the measured divergence statistics reported alongside.

The shipping package also has a product-loading one-prompt generation
certification on the 3,844-token speed anchor. Five fresh-process
automatic-DSpark arms each reproduced the complete 38-token plain rail, ran 13
fixed-three rounds, proposed 39 tokens, accepted 25, and used no plain fallback.
This establishes pooled-target generation and DSpark interaction for that
prompt. It does not make cross-lattice token identity a general contract;
broader real-package quality-gate runs through the speculative path remain
pending.

## Public and private fixture boundary

Public and committed:

- Q0 renderer/tokenizer vectors copied from the public ds4 suite;
- Q1 prompts, selected tokens, and top-20 candidate-set records from that suite;
- Q2 prompts without provider continuations;
- the deterministic Q3 generator, manifest, and answer set;
- harness code and synthetic/self-verifying fixtures.

Private and ignored:

- API-captured continuations, selected-token records, and top-logprob payloads,
  whether they serve Q1 or Q2. Material the upstream suite published stays
  public; material this project captured from a provider does not;
- Q4 teacher and candidate logit dumps, and the marker set they are scored
  against;
- the WikiText corpus, which is third-party text referenced by digest and never
  copied in;
- unpublished benchmark question prose, answer keys, or provider responses;
- refreshed oracle dumps and one-off investigation payloads;
- credentials and any local paths that identify their storage location.

The public docs explain how each gate works without embedding the private
oracle. Do not commit an API-derived answer merely because a local comparison
used it.

## When each gate is required

- Renderer or tokenizer changes: Q0, then the gates affected by changed token
  input.
- Package recipe or quantized-math changes: Q1, Q2, Q3, Q4, and the WikiText
  perplexity gate. The perplexity arm is not optional here: it is the only
  instrument that has caught a numeric blowup the numbered gates all scored as
  finite.
- Cache, attention, indexer, or long-prefill changes: Q1 and Q3, plus Q2 when
  floating-point evaluation order changes.
- Speculative decoding changes: the replay identity harness
  (`moespresso-ds4-dspark-replay`), then Q1 and Q3 through the speculative
  path.
- Pure performance changes: the cheapest gate that proves the intended route,
  followed by the full affected family gate before promotion.

Run `make lint` and `make test` in addition to these manual gates when code
changes touch the runtime path.
