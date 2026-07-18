"""Optimize: turn probe evidence + constraints into an `optimizer_decision` artifact.

Activation-weighted fidelity F + opt-in worst-layer CVaR tail constraint,
deterministic, with explicit INFEASIBLE results. The converter consumes the
decision; the runtime never reinterprets it.
"""
