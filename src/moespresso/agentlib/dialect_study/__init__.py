"""Ornith tool-call dialect study.

A fixed set of short agentic episodes is executed for real against a served
model, once per dialect arm, judged by a scripted judge on parse rate, call
correctness, task success, and tokens spent. The arms:

- ``native``: request-level tools rendered by the vendored Qwen template;
  the model emits ``<tool_call><function=...>`` XML parsed by
  ``agentlib.qwenxml``.
- ``envelope``: the Terminus-2-style JSON action envelope in plain text,
  parsed by ``agentlib.envelope``.
- ``dsml``: DeepSeek DSML text markers taught through the system prompt,
  parsed by ``agentlib.dsml``.

The runner attaches to an already-running server and coordinates GPU use
through the shared measurement lock. ``tests/test_dialect_study.py`` drives
the whole loop with a fake client, so the harness is pinned without a GPU.
"""
