"""Inventory: turn a source model into a checked, typed `source_inventory` artifact.

Maps raw source tensor names to internal roles ONCE (the resolve-once boundary),
records shape/dtype/status, validates against the architecture's expected
tensors. Runnable even on models that will never be converted. No other phase
re-derives tensor names.
"""
