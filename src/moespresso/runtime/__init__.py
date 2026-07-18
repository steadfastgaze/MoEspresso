"""Runtime: load a package manifest and execute, no source archaeology.

Validates the manifest (version, architecture, required ops), loads tensors per
declared formats/residency, runs the model graph. The strict `mjtq` format + MoE
first; reuses the jang TurboQuant codec at the compute layer. Model-specific
*execution* code is allowed; model-specific *detection/conversion* is not.
"""
