.PHONY: help install test lint fmt lock lock-check dist-check clean roadtest

help:
	@echo "make install        - sync the uv environment (incl. dev group)"
	@echo "make test           - run the full test suite (lock-strict)"
	@echo "make lint           - ruff check (lock-strict)"
	@echo "make fmt            - ruff format"
	@echo "make lock           - re-resolve uv.lock after a deliberate dep change"
	@echo "make lock-check     - fail if uv.lock is stale vs pyproject.toml"
	@echo "make dist-check     - build and audit the public wheel and sdist"
	@echo "make roadtest       - cumulative-session engine road-test against a"
	@echo "                      really served package (opt-in, GPU-bound; set"
	@echo "                      MOESPRESSO_ROADTEST_PACKAGE or ROADTEST_ARGS)"
	@echo "make clean          - remove caches"

install:
	uv sync

test:
	uv run --locked python -m pytest

lint:
	uv run --locked ruff check src tests

fmt:
	uv run ruff format src tests

# Engine road-test: a scripted cumulative agentic session against a really
# served model, asserting cache events, disk checkpoints, and restart resume
# turn over turn. Opt-in and GPU-bound; never part of `make test`. The runner
# itself is a pure HTTP client that launches the served process.
roadtest:
	uv run --locked python -m moespresso.agentlib.roadtest $(ROADTEST_ARGS)

# Deliberate "I changed deps, re-resolve" step.
lock:
	uv lock

# Standalone gate (read-only, no env build): is the lockfile current?
lock-check:
	uv lock --check

# Build both public artifacts in a temporary directory, then reject private
# material, machine-local paths, missing licenses, and missing product surfaces.
dist-check:
	uv run --locked python tests/test_distribution_audit.py --build

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
