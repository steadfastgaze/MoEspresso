# Frozen 48-question benchmark

The reported MoEspresso quality comparison uses generated answers to 48 public
LiveBench questions across six categories, scored with the official task
scorers. This benchmark is separate from the official LiveBench leaderboard.

## Setup

Requires Python 3.11 or 3.12, [uv](https://docs.astral.sh/uv/), and Docker.

```sh
uv sync
uv run python -c 'from common import load_questions; print(len(load_questions()))'
docker build -t livebench-short-grader:1263ee4 .
```

The second command downloads the pinned public datasets, verifies their hashes,
and derives the questions by the lowest salted SHA-256 rank within each fixed
category/task quota. No question-ID list is stored in this kit.

## Run MoEspresso

Start the server separately with `MOESPRESSO_DISK_KV=off`, then substitute its
advertised model ID:

```sh
uv run python run.py \
  --base-url http://127.0.0.1:8080/v1 --model MODEL_ID \
  --out runs/local --max-output-tokens 131072 --reasoning-effort medium \
  --require-capacity 223 --require-cache-prior --require-package
```

The three `--require-*` options reject a server with the wrong package, expert
capacity, or Cache-Prior 2/2 policy.

## Run another API

Set `OPENAI_API_KEY`, then run:

```sh
uv run python run.py \
  --base-url https://API_HOST/v1 --model MODEL_ID --out runs/remote \
  --sampling provider --max-output-tokens 128000 \
  --reasoning-effort medium --stream
```

Use `--api-key-env NAME` for another environment variable. If an API accepts the
fixed controls, omit `--sampling provider` to send temperature 0, top-p 1,
top-k 0, min-p 0, and presence penalty 0.

## Resume and score

Repeat the same generation command to resume, skipping completed questions.
If an interrupted request has an uncertain outcome, the run stops without
resending it. Changed settings require a new output directory.

```sh
uv run python score.py --run runs/local --out runs/local-scores.json
```

Scoring requires all 48 answers and gives zero to output-cap hits. Each answer
is graded in a fresh, network-disabled, read-only container; generated code
never runs on the host.

## Protocol

Each of the six categories contains two tasks with four questions per task.
Every question is one user message with no system prompt, tools, or conversation
state. Thinking runs at medium effort, with output ceilings of 131,072 tokens
locally and 128,000 tokens for APIs. Full responses, reasoning, usage, cost,
request settings, and server identity are retained. The run directory contains
the questions, keys, and complete responses.

## Check the kit

```sh
uv run pytest -q
uv run ruff check .
TEST_GRADING_IMAGE=1 uv run pytest -q
```

The last command exercises all twelve scorers through the Docker image.
