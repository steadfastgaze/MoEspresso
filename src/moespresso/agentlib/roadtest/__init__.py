"""Engine road-test: a long cumulative agentic session with counter assertions.

The road-test drives a really served model through agentlib for dozens of
turns, executes the tool calls the model makes on a generated fixture
workspace, and asserts the engine's cache evidence after every request:
``usage.prompt_cache`` events, ``cached_tokens`` growth, disk checkpoint
writes at 256-aligned frontiers, and the ``/health`` cumulative counters. It
manages the server lifecycle itself, including mid-session restarts that must
resume from disk checkpoints, and an interleaved second session under its own
cache key. Opt-in via ``make roadtest``; never part of ``make test``.

Modules:

- ``fixture``: deterministic fixture workspace generation.
- ``script``: the scripted user side of the session.
- ``ledger``: per-session counter bookkeeping and assertion helpers.
- ``server``: served-process lifecycle plus shared-GPU coordination.
- ``__main__``: the run driver (``python -m moespresso.agentlib.roadtest``).
"""
