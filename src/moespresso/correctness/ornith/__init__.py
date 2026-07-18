"""Ornith-1.0-35B quality gate v2.

Model-specific quality gates for the Ornith Qwen 3.5 35B A3B fine-tune. The gate
runs three instrument families against the served package at the model's
recommended sampling profile with thinking off: exact-answer hard reasoning,
self-verifying agentic coding (the harness executes the model's emitted code
against hidden tests in a sandboxed subprocess), and exact-scored long-context
recall over real repository text. Every item carries a fail-class label
(clean-pass, pass-at-cap, fail-genuine, fail-truncated), a per-item seed, and a
token budget sized from a measured run.

Shared correctness-root helpers stay at the root; this subpackage holds the
Ornith-specific task content and scoring. It mirrors the correctness-tree layout
that keeps DeepSeek-V4 gates under `correctness.deepseek_v4`.
"""
