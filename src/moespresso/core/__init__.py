"""Core: the artifact contract that every phase obeys.

`artifact.py` owns the baseline rules (schema_version, content-hash artifact_id,
producer/inputs/subject/status/validation). Everything else in MoEspresso is a
producer or consumer of these artifacts.
"""
