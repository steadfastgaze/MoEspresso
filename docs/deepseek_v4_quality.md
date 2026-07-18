# DeepSeek-V4-Flash quality gates

DeepSeek-V4-Flash has four model-specific quality gates. They are manual real
package runs: they do not run in `make test`, they do not run in CI, and they
never discover a package implicitly.

The gates answer different questions. Passing a higher-numbered gate does not
replace the lower gates, and package verification does not replace any of them.

## Run the gates

Pass `--package` explicitly or set `MOESPRESSO_DS4_QUALITY_PACKAGE`:

```bash
uv run --locked moespresso-ds4-quality \
  q0 --package <package-dir>

uv run --locked moespresso-ds4-quality \
  q1 --package <package-dir>

uv run --locked moespresso-ds4-quality \
  q2 --package <package-dir> --reference <private-reference.json>

uv run --locked moespresso-ds4-quality \
  q3 --package <package-dir>
```

Q0, Q1, Q2, and Q3 accept `--json-out <path>`. Each run records the installed
MLX wheel tag because different wheel builds of one MLX version can form
different, internally deterministic numeric lattices.

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

## Q1: official selected-token and top-20 parity

Q1 serves five committed public prompts with greedy decoding, temperature 0,
thinking disabled, and a local top-20 capture. Across those prompts the
reference contains 17 selected generation steps.

For every step, the current contract requires:

- the official selected token to be the local rank-0 token;
- at least one token of overlap between the official and local top-20 sets;
- the expected prompt, render, model, and run metadata.

The copied official records use `-9999.0` sentinels for non-selected
alternatives. Those values are not calibrated log probabilities, so Q1 does not
treat their numeric deltas as evidence.

The recorded byte-faithful package reached 17/17 on one MLX wheel lattice in
repeated runs. Another wheel lattice reached 16/17 with one deterministic
casing knife-edge. The current strict Q1 gate requires rank 0 at every step, so
the latter result is documented environment sensitivity rather than a passing
substitute.

Q1 catches prompt drift and serving errors large enough to move short and
medium continuations. It samples the distribution and does not prove full-logit
identity.

## Q2: target-token negative log likelihood

Q2 teacher-forces provider continuations over the tracked 100-prompt set and
measures the negative log likelihood assigned by the local package. It is the
continuous comparison instrument for kernel numerics, package realizations,
and low-bit recipe changes.

Create or refresh a local reference with:

```bash
uv run --locked moespresso-ds4-quality \
  q2-capture --out <private-reference.json>
```

The capture command reads `OPENROUTER_TOKEN` from the environment or `.env` and
does not write the token into the payload. The resulting continuations and
top-logprob records are API-derived oracle material and must remain outside the
committed public tree.

Compare two Q2 score artifacts with:

```bash
uv run --locked moespresso-ds4-quality \
  q2-compare <old-q2.json> <new-q2.json>
```

Q2 reports aggregate and per-case NLL plus case wins. It has no universal
package-independent threshold; compare packages on the same prompt/reference
set and MLX wheel. The two recorded wheel lattices anchor at
`0.37653323427558466` and `0.378282373753177`, each reproducible within its own
environment.

## Q3: deterministic long-context fact recall

Q3 generates a deterministic long story with 16 embedded name-to-number facts
and requires exact `Name=number` answer lines. The fixture generator, manifest,
and answers are public and API-free.

Q3 exercises the compressed-attention, indexer, long-prefill, and cache paths
that the shorter Q1 prompts barely touch. The accepted byte-faithful package
scores 16/16.

## Public and private fixture boundary

Public and committed:

- Q0 renderer/tokenizer vectors copied from the public ds4 suite;
- Q1 prompts, selected tokens, and top-20 candidate-set records from that suite;
- Q2 prompts without provider continuations;
- the deterministic Q3 generator, manifest, and answer set;
- harness code and synthetic/self-verifying fixtures.

Private and ignored:

- Q2 API-captured continuations and top-logprob payloads;
- unpublished benchmark question prose, answer keys, or provider responses;
- refreshed oracle dumps and one-off investigation payloads;
- credentials and any local paths that identify their storage location.

The public docs explain how each gate works without embedding the private
oracle. Do not commit an API-derived answer merely because a local comparison
used it.

## When each gate is required

- Renderer or tokenizer changes: Q0, then the gates affected by changed token
  input.
- Package recipe or quantized-math changes: Q1, Q2, and Q3.
- Cache, attention, indexer, or long-prefill changes: Q1 and Q3, plus Q2 when
  floating-point evaluation order changes.
- Pure performance changes: the cheapest gate that proves the intended route,
  followed by the full affected family gate before promotion.

Run `make lint` and `make test` in addition to these manual gates when code
changes touch the runtime path.
