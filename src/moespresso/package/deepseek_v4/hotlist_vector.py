"""Vendored cold-start expert prewarm ranking for DeepSeek-V4 packages.

The DS4 build imatrix is the legacy .dat format, which stores per-tensor
activation sums with a single call counter and carries no per-expert routing
counts, so the standard imatrix-counts hotlist cannot be built from it. The
ranking vendored here was extracted once from a GGUF-format community imatrix
whose expert counts align exactly with the byte-faithful package layout
(43 routed layers, 256 experts per layer); provenance (source name and
sha256) is recorded in the payload itself. That imatrix contributes nothing
else: quantization decisions, calibration vectors, and fit checks all keep
the recorded build imatrix.

The package builder consumes this through
`hotlist.write_package_expert_hotlist_from_payload`, which re-validates the
layer alignment against the package being built, and only when the build
imatrix yields no expert counts of its own.
"""

from __future__ import annotations

import json
from importlib.resources import files

VECTOR_NAME = "expert_hotlist_vector.json"


def load_vendored_expert_hotlist() -> dict:
    """The committed DS4 expert prewarm ranking payload."""
    text = files(__package__).joinpath("data", VECTOR_NAME).read_text(
        encoding="utf-8")
    return json.loads(text)
