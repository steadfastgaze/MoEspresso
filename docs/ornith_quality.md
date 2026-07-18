# Ornith quality gate

Ornith uses a model-specific nine-item served gate. This manual real-package run
is invoked only through `moespresso-ornith-gate`; pytest never starts it.

The gate combines exact reasoning answers, sandboxed self-verifying code, and
long-context retrieval. All three instrument families run through the real
manifest loader and the same rendering/generation seam used by the HTTP server.

## Run the gate

```bash
uv run --locked moespresso-ornith-gate \
  <package-dir> \
  --json-out <ornith-gate.json>
```

The command exits 0 only when every selected item passes. Restrict a diagnostic
run with a comma-separated subset:

```bash
uv run --locked moespresso-ornith-gate \
  <package-dir> \
  --families hard_reasoning,agentic_coding,long_context
```

`--sandbox-timeout <seconds>` controls the per-task subprocess timeout for the
coding family. A subset is useful for diagnosis; the promotion gate is the full
nine-item run.

## Fixed execution profile

Every item runs with thinking off and the model's recorded sampling profile:

| Parameter | Value |
|---|---:|
| temperature | 1.0 |
| top_p | 0.95 |
| top_k | 20 |
| min_p | 0.0 |
| presence_penalty | 1.5 |

Each item also carries a stable random seed and a measured token budget. The
report records the package manifest id, package family, selected instrument
families, per-item wall time, completion-token count, cap status, and aggregate
pass count.

## Four hard-reasoning items

The reasoning family selects four exact-answer questions from a code-verified
private benchmark key. It covers integer and fraction answer forms and includes
at most one deliberately token-hungry item.

Scoring extracts the stated answer, normalizes integers and reduced fractions
without numeric tolerance, and compares the result exactly with the key. The
question prose and answer key are loaded by path from an ignored private fixture
directory and are never embedded in the public task definitions or reports.

## Three agentic coding items

The coding family is public and self-verifying. Each item asks the model to
submit a small Python implementation through the packaged tool-call dialect.
The harness extracts the code and executes hidden test cases in a sandboxed
subprocess with a timeout.

A coding item passes only when code was extracted and every hidden test passes.
The public tasks cover run-length encoding, bracket balance, and word-frequency
selection. They are original fixtures rather than unpublished benchmark
material.

## Two long-context items

The long-context family builds a reproducible corpus from committed repository
source and plants public facts inside it. One item asks for a single needle; the
other asks for an aggregate over three planted values.

Answers are normalized and exact-scored. This family exercises prompt
rendering, long prefill, cache state, and retrieval without an external API or a
private answer source.

## Failure classes

Every row receives one outcome class:

- `clean-pass`: correct and stopped before the token cap;
- `pass-at-cap`: correct but ended at the cap;
- `fail-genuine`: a wrong or otherwise scorable failed answer;
- `fail-truncated`: hit the cap without a scorable answer.

The distinction matters when changing token budgets, but promotion still
requires every item to pass.

## Current acceptance record

The byte-faithful 19.91 GiB package passed gate v2 at 9/9 in the full-resident
configuration. The same gate passed 9/9 at the streamed cap-192 operating point
used for the 32 GB-class record.

Those results establish the two tested runtime modes for that package and
environment. A new package, MLX wheel, template, sampling change, or math path
requires a fresh full gate; the old count is not transferable by name or size.

## Public and private boundary

Public and committed:

- the gate runner, exact normalization, fail-class logic, and sandbox;
- the three coding tasks and their hidden self-verifying cases;
- the long-context corpus builder, planted facts, questions, and answers;
- seeds, token budgets, sampling profile, and report schema.

Private and ignored:

- unpublished reasoning question prose and answer keys;
- provider or benchmark responses captured during investigation;
- generated answer dumps that reproduce private source material;
- credentials and machine-local fixture paths.

Reports should contain scores and bounded diagnostic fields while excluding
private question prose and keys. Do not publish a benchmark answer merely because the
harness can score it locally.

## When to run

Run the full gate for changes to:

- package bytes, quantization, or K-quant installation;
- tokenizer, chat template, thinking policy, or tool-call rendering;
- model graph math, attention, recurrent state, routing, or expert execution;
- in-memory or disk cache behavior;
- resident/streamed scheduling or memory-cap logic;
- the gate profile, task definitions, scoring, or sandbox.

Performance-only work may use a smaller diagnostic family during iteration, but
must return to the full 9/9 gate before promotion. Also run `make lint` and
`make test` for runtime-path code changes.
