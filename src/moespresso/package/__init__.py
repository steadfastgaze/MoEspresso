"""Package: write tensors + an explicit `package_manifest` per the package plan.

The manifest declares to the engine exactly how to run (architecture, tensor
roles, weight formats/transforms (the strict `mjtq` format first, via the jang
TurboQuant codec), expert-selection capability, residency, required backend ops)
so the runtime never guesses. The converter is deliberately boring: read a
package plan, write tensors + manifest.
"""
