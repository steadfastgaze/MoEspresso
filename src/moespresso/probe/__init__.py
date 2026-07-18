"""Probe: measure tensor sensitivity into a `probe_evidence` artifact.

Consumes inventory roles (never raw names): activation-weighted reconstruction
quality (per-channel imatrix h), streaming, fp16. Emits per-(bits,gs) quality
tables the optimizer consumes.
"""
